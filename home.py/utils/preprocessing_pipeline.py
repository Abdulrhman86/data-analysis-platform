import pandas as pd
import numpy as np
import streamlit as st
import json
import pickle
import base64
import difflib
import functools
from datetime import datetime


def record_step(processor_type):
    """
    Decorator to record preprocessing steps in the pipeline.
    Apply this to processor methods to automatically record steps.

    Args:
        processor_type: Type of processor ('numeric', 'categorical', etc.)
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, df, *args, **kwargs):
            # Extract and remove special parameter that controls recording
            record_to_pipeline = kwargs.pop('record_to_pipeline', False)

            # Apply the function
            result = func(self, df, *args, **kwargs)

            # Only record if explicitly requested AND a pipeline exists
            if record_to_pipeline and 'pipelines' in st.session_state and st.session_state.current_pipeline is not None:
                # Extract function name
                function_name = func.__name__

                # Build parameters dictionary from args and kwargs
                params = kwargs.copy()
                # Add first argument as 'column' if not already in kwargs
                if args and 'column' not in params and len(args) > 0:
                    params['column'] = args[0]

                # Create description based on function and parameters
                if 'column' in params:
                    description = f"{function_name} on {params['column']}"
                else:
                    description = f"{function_name} on multiple columns"

                # Record the step
                try:
                    record_preprocessing_step(
                        processor_type=processor_type,
                        function_name=function_name,
                        params=params,
                        description=description
                    )
                except Exception as e:
                    # Log error but don't prevent function from working
                    print(f"Error recording step: {str(e)}")

            return result

        # Mark wrapped methods so pipeline replay can confirm a step's function
        # is a real preprocessing transform (not an arbitrary method reached via
        # a crafted/imported pipeline JSON).
        wrapper._is_preprocessing_step = True
        return wrapper

    return decorator

class PreprocessingStep:
    """Class representing a single preprocessing step"""

    def __init__(self, name, processor_type, function_name, params, description=None, depends_on=None):
        """
        Initialize a preprocessing step

        Args:
            name: Name of the step
            processor_type: Type of processor (e.g., 'numeric', 'categorical')
            function_name: Name of the function to call
            params: Dictionary of parameters
            description: Optional description
            depends_on: List of step indices this step depends on
        """
        self.name = name
        self.processor_type = processor_type
        self.function_name = function_name
        self.params = params
        self.description = description or f"{name} ({function_name})"
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.depends_on = depends_on or []  # Dependencies tracking

    def apply(self, df, processors):
        """
        Apply the preprocessing step to a DataFrame with improved error handling

        Args:
            df: pandas DataFrame
            processors: Dictionary of processor instances

        Returns:
            Transformed DataFrame
        """
        # Get the appropriate processor
        processor = processors.get(self.processor_type)
        if not processor:
            available_processors = list(processors.keys())
            raise ValueError(
                f"Processor type '{self.processor_type}' not found. Available processors: {available_processors}"
            )

        # Get the function
        func = getattr(processor, self.function_name, None)
        if not func:
            processor_methods = [method for method in dir(processor)
                                 if callable(getattr(processor, method)) and not method.startswith('_')]
            raise ValueError(
                f"Function '{self.function_name}' not found in {self.processor_type} processor. "
                f"Available methods: {processor_methods[:10]}"
            )

        # Security: only allow functions explicitly marked as preprocessing steps
        # (see @record_step). Prevents a crafted/imported pipeline JSON from
        # invoking arbitrary processor methods during replay.
        if not getattr(func, '_is_preprocessing_step', False):
            raise ValueError(
                f"Function '{self.function_name}' is not an allowed preprocessing step."
            )

        # Validate required columns exist
        missing_columns = []
        for param_name, param_value in self.params.items():
            if param_name in ['column', 'column1', 'column2', 'group_col', 'agg_col', 'date_col', 'value_col'] and \
                    isinstance(param_value, str) and param_value not in df.columns:
                missing_columns.append(param_value)
            # Check for list of columns
            elif isinstance(param_value, list) and param_name in ['columns', 'group_by']:
                for col in param_value:
                    if isinstance(col, str) and col not in df.columns:
                        missing_columns.append(col)

        if missing_columns:
            available_cols = df.columns.tolist()
            preview = available_cols[:5] + (["..."] if len(available_cols) > 5 else [])
            raise ValueError(
                f"Required columns not found in dataset: {missing_columns}. Available columns: {preview}"
            )

        try:
            # Apply the function with parameters
            result = func(df, **self.params)

            # Validate result
            if result is None:
                raise RuntimeError(
                    f"Function {self.function_name} returned None. Expected a DataFrame."
                )

            if not isinstance(result, pd.DataFrame):
                raise RuntimeError(
                    f"Function {self.function_name} returned {type(result).__name__}. Expected a DataFrame."
                )

            return result

        except Exception as e:
            # Enhance error messages with potential causes and more details
            error_msg = str(e)

            if "KeyError" in error_msg:
                col_name = error_msg.replace("KeyError: ", "").replace("'", "").replace('"', '')
                raise RuntimeError(
                    f"Column '{col_name}' not found. Please check if column exists in dataset or update step parameters. "
                    f"Available columns: {list(df.columns[:5])}"
                )

            elif "TypeError" in error_msg:
                # Provide more information about the type mismatch
                if "dtype" in error_msg:
                    raise RuntimeError(
                        f"Type mismatch error: {error_msg}. Check if column data types match function requirements. "
                        f"Consider using type conversion before this step."
                    )
                else:
                    param_issue = any(param in error_msg for param in self.params.keys())
                    if param_issue:
                        raise RuntimeError(
                            f"Parameter error in {self.function_name}: {error_msg}. "
                            f"Check if the parameters provided are correct."
                        )
                    else:
                        raise RuntimeError(
                            f"Type error in {self.function_name}: {error_msg}. "
                            f"Check if function inputs match expected types."
                        )

            elif "ValueError" in error_msg:
                if "could not convert" in error_msg.lower():
                    raise RuntimeError(
                        f"Value conversion error: {error_msg}. "
                        f"Consider checking data types or preprocessing the column first."
                    )
                else:
                    raise RuntimeError(f"Value error in {self.function_name}: {error_msg}")

            elif "ZeroDivisionError" in error_msg:
                raise RuntimeError(
                    f"Division by zero in {self.function_name}. "
                    f"Consider adding a small value or filtering zeros before division."
                )

            elif "AttributeError" in error_msg:
                raise RuntimeError(
                    f"Attribute error in {self.function_name}: {error_msg}. "
                    f"This may be caused by incorrect data type or missing attribute."
                )
            else:
                # For any other error, provide context
                raise RuntimeError(f"Error in {self.function_name}: {error_msg}")

    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'processor_type': self.processor_type,
            'function_name': self.function_name,
            'params': self.params,
            'description': self.description,
            'timestamp': self.timestamp,
            'depends_on': self.depends_on
        }

    @classmethod
    def from_dict(cls, data):
        """Create a PreprocessingStep from a dictionary"""
        step = cls(
            name=data['name'],
            processor_type=data['processor_type'],
            function_name=data['function_name'],
            params=data['params'],
            description=data.get('description'),
            depends_on=data.get('depends_on', [])
        )
        step.timestamp = data.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        return step


class PreprocessingPipeline:
    """Class representing a sequence of preprocessing steps"""

    def __init__(self, name="Preprocessing Pipeline"):
        """
        Initialize a preprocessing pipeline

        Args:
            name: Name of the pipeline
        """
        self.name = name
        self.steps = []
        self.creation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def add_step(self, step):
        """
        Add a preprocessing step to the pipeline

        Args:
            step: PreprocessingStep object
        """
        self.steps.append(step)

    def remove_step(self, index):
        """
        Remove a step from the pipeline

        Args:
            index: Index of the step to remove
        """
        if 0 <= index < len(self.steps):
            # Update dependencies for steps that depended on the removed step
            for step in self.steps:
                # Remove the index from dependencies
                if index in step.depends_on:
                    step.depends_on.remove(index)

                # Adjust dependencies that pointed to steps after the removed one
                step.depends_on = [dep if dep < index else dep - 1 for dep in step.depends_on]

            # Remove the step
            del self.steps[index]

    def map_columns(self, original_columns, new_columns):
        """
        Create a mapping between original dataset columns and new dataset columns

        Args:
            original_columns: List of column names from the original dataset
            new_columns: List of column names from the new dataset

        Returns:
            Dictionary mapping original column names to new column names
        """
        column_map = {}

        # First try exact matches
        for orig_col in original_columns:
            if orig_col in new_columns:
                column_map[orig_col] = orig_col

        # For unmatched columns, try fuzzy matching by name
        unmatched_orig_cols = [col for col in original_columns if col not in column_map]
        unmatched_new_cols = [col for col in new_columns if col not in column_map.values()]

        if unmatched_orig_cols and unmatched_new_cols:
            for orig_col in unmatched_orig_cols:
                # Find closest match by name similarity
                best_match = None
                best_score = 0

                for new_col in unmatched_new_cols:
                    # Simple similarity: Percentage of shared characters
                    similarity = difflib.SequenceMatcher(None, orig_col.lower(), new_col.lower()).ratio()

                    if similarity > 0.7 and similarity > best_score:  # Threshold of 70% similarity
                        best_score = similarity
                        best_match = new_col

                if best_match:
                    column_map[orig_col] = best_match
                    unmatched_new_cols.remove(best_match)

        return column_map

    def apply(self, df, processors, column_map=None):
        """
        Apply all preprocessing steps to a DataFrame with support for different datasets

        Args:
            df: pandas DataFrame
            processors: Dictionary of processor instances
            column_map: Optional mapping of original column names to new column names

        Returns:
            Transformed DataFrame
        """
        result_df = df.copy()

        # Automatically create column mapping if not provided
        if column_map is None:
            # Extract column names from step parameters
            original_columns = set()
            for step in self.steps:
                for param_name, param_value in step.params.items():
                    if param_name in ['column', 'column1', 'column2'] and isinstance(param_value, str):
                        original_columns.add(param_value)

            # Create mapping
            column_map = self.map_columns(list(original_columns), df.columns.tolist())

            # Show mapping info if columns were mapped
            mapped_cols = {k: v for k, v in column_map.items() if k != v}
            if mapped_cols and len(original_columns) > 0:
                st.info(f"Mapped {len(mapped_cols)} column(s) to match this dataset")
                if len(mapped_cols) <= 5:  # Show mapping details if not too many
                    for orig, new in mapped_cols.items():
                        st.write(f"• '{orig}' → '{new}'")

        # Track which steps have been applied
        applied_steps = set()

        # Keep trying to apply steps until we can't make progress
        progress_made = True
        step_errors = {}  # Track errors
        dependency_issues = {}  # Track dependency issues

        # First validate dependencies to avoid circular dependencies
        dependency_graph = {i: step.depends_on for i, step in enumerate(self.steps)}
        try:
            self._validate_dependencies(dependency_graph)
        except ValueError as e:
            st.error(f"Invalid pipeline configuration: {str(e)}")
            return result_df

        while progress_made:
            progress_made = False

            for i, step in enumerate(self.steps):
                # Skip already applied steps
                if i in applied_steps:
                    continue

                # Check if dependencies are satisfied
                dependencies_met = all(dep in applied_steps for dep in step.depends_on)
                if not dependencies_met:
                    missing_deps = [dep + 1 for dep in step.depends_on if dep not in applied_steps]
                    dependency_issues[i] = f"Waiting for steps {missing_deps}"
                    continue  # Skip steps with unmet dependencies

                try:
                    # Create a copy of step parameters with mapped column names
                    mapped_params = step.params.copy()

                    # Update column parameters based on mapping
                    column_params = ['column', 'column1', 'column2', 'x_column', 'y_column',
                                     'group_col', 'agg_col', 'date_col', 'value_col']

                    for param in column_params:
                        if param in mapped_params and isinstance(mapped_params[param], str) and mapped_params[
                            param] in column_map:
                            mapped_params[param] = column_map[mapped_params[param]]

                    # Handle list parameters (e.g., columns, group_by)
                    for param, value in mapped_params.items():
                        if isinstance(value, list) and all(isinstance(item, str) for item in value):
                            mapped_params[param] = [column_map.get(col, col) for col in value]

                    # Create a temporary step with mapped parameters
                    temp_step = PreprocessingStep(
                        name=step.name,
                        processor_type=step.processor_type,
                        function_name=step.function_name,
                        params=mapped_params,
                        description=step.description
                    )

                    # Apply the step
                    result_df = temp_step.apply(result_df, processors)

                    # Mark this step as applied and remove any dependency issues
                    applied_steps.add(i)
                    if i in dependency_issues:
                        del dependency_issues[i]
                    progress_made = True

                except Exception as e:
                    # Record the error with more context
                    context = f"Step {i + 1} ({step.name}, {step.function_name})"
                    error_msg = str(e)
                    step_errors[i] = f"{context}: {error_msg}"

            # If we've gone through all steps without progress, we're either done or have issues
            if not progress_made and len(applied_steps) < len(self.steps):
                unapplied = [i + 1 for i in range(len(self.steps)) if i not in applied_steps]

                # Create an error summary
                if step_errors or dependency_issues:
                    st.error(f"Could not apply {len(unapplied)} step(s) due to errors")

                    # Show dependency issues first
                    if dependency_issues:
                        st.warning("### Dependency Issues:")
                        for i, issue in sorted(dependency_issues.items()):
                            st.warning(f"Step {i + 1} ({self.steps[i].name}): {issue}")

                    # Then show execution errors
                    if step_errors:
                        st.error("### Execution Errors:")
                        for i, error in sorted(step_errors.items()):
                            st.error(error)

                        # Suggest potential fixes for common errors
                        self._suggest_error_fixes(step_errors, result_df)
                else:
                    st.warning(f"Could not apply steps {unapplied} due to unknown issues")
                break

        return result_df

    def _validate_dependencies(self, dependency_graph):
        """
        Validate that there are no circular dependencies in the pipeline

        Args:
            dependency_graph: Dictionary mapping step indices to their dependencies

        Raises:
            ValueError if circular dependencies are detected
        """
        visited = set()
        temp_visited = set()

        def check_cycle(node):
            if node in temp_visited:
                # We found a cycle
                return True
            if node in visited:
                # Already checked, no cycle found
                return False

            temp_visited.add(node)

            # Check all dependencies
            for dep in dependency_graph.get(node, []):
                if check_cycle(dep):
                    return True

            # Remove from temporary visited
            temp_visited.remove(node)
            # Mark as fully visited
            visited.add(node)
            return False

        # Check each node
        for node in dependency_graph:
            if check_cycle(node):
                cycle_nodes = list(temp_visited)
                cycle_str = " → ".join([str(n + 1) for n in cycle_nodes]) + f" → {cycle_nodes[0] + 1}"
                raise ValueError(f"Circular dependency detected in pipeline steps: {cycle_str}")

    def _suggest_error_fixes(self, step_errors, df):
        """
        Suggest potential fixes for common errors

        Args:
            step_errors: Dictionary of step errors
            df: Current DataFrame
        """
        suggestions = []

        for i, error in step_errors.items():
            error_msg = error.lower()

            # Column not found errors
            if "column" in error_msg and "not found" in error_msg:
                suggestions.append("- Check for typos in column names or ensure the required columns exist")
                suggestions.append(
                    f"- Available columns: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")

            # Type mismatch errors
            if "type" in error_msg and ("mismatch" in error_msg or "error" in error_msg):
                suggestions.append("- Ensure columns have the correct data types for the operation")
                suggestions.append("- Use type conversion functions before applying operations")

            # Division by zero
            if "division by zero" in error_msg or "divide by zero" in error_msg:
                suggestions.append("- Handle division by zero cases by adding a small value or filtering data")

            # General error suggestion
            if not suggestions:
                suggestions.append("- Review step parameters and ensure they match the current data")
                suggestions.append("- Try breaking complex steps into simpler steps")

        if suggestions:
            st.info("### Suggestions to fix errors:")
            for suggestion in set(suggestions):  # Remove duplicates
                st.write(suggestion)

    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'creation_date': self.creation_date,
            'steps': [step.to_dict() for step in self.steps]
        }

    @classmethod
    def from_dict(cls, data):
        """Create a PreprocessingPipeline from a dictionary"""
        pipeline = cls(name=data['name'])
        pipeline.creation_date = data.get('creation_date', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        for step_data in data['steps']:
            pipeline.add_step(PreprocessingStep.from_dict(step_data))

        return pipeline

    def to_json(self):
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str):
        """Create a PreprocessingPipeline from a JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)


def pipeline_manager_ui(st, df, processors):
    """
    Streamlit UI for managing preprocessing pipelines

    Args:
        st: Streamlit instance
        df: pandas DataFrame
        processors: Dictionary of processor instances

    Returns:
        DataFrame if pipeline was applied, otherwise None
    """
    st.subheader("Preprocessing Pipeline Manager")

    # Initialize session state for pipelines
    if 'pipelines' not in st.session_state:
        st.session_state.pipelines = {}

    if 'current_pipeline' not in st.session_state:
        st.session_state.current_pipeline = None

    # Check if there's a modified DataFrame
    result_df = None

    # Tabs for Pipeline Management
    pipeline_tab, step_config_tab, export_tab, import_tab = st.tabs([
        "Manage Pipelines", "Edit Steps", "Export Pipeline", "Import Pipeline"
    ])

    with pipeline_tab:
        # Pipeline selection or creation
        col1, col2 = st.columns([3, 1])

        with col1:
            pipeline_options = list(st.session_state.pipelines.keys())
            pipeline_options.insert(0, "-- Create New Pipeline --")

            selected_pipeline = st.selectbox(
                "Select Pipeline:",
                pipeline_options
            )

            if selected_pipeline == "-- Create New Pipeline --":
                new_pipeline_name = st.text_input("New Pipeline Name:", "My Pipeline")

                if st.button("Create Pipeline"):
                    if new_pipeline_name in st.session_state.pipelines:
                        st.warning(f"Pipeline '{new_pipeline_name}' already exists.")
                    else:
                        st.session_state.pipelines[new_pipeline_name] = PreprocessingPipeline(new_pipeline_name)
                        st.session_state.current_pipeline = new_pipeline_name
                        st.success(f"Created new pipeline: {new_pipeline_name}")
                        st.rerun()
            else:
                st.session_state.current_pipeline = selected_pipeline

        with col2:
            if st.session_state.current_pipeline in st.session_state.pipelines:
                if st.button("Delete Pipeline"):
                    del st.session_state.pipelines[st.session_state.current_pipeline]
                    st.session_state.current_pipeline = None
                    st.success("Pipeline deleted.")
                    st.rerun()

        # If a pipeline is selected, show its steps and controls
        if st.session_state.current_pipeline in st.session_state.pipelines:
            pipeline = st.session_state.pipelines[st.session_state.current_pipeline]

            st.write(f"### Pipeline: {pipeline.name}")
            st.write(f"Created: {pipeline.creation_date}")
            st.write(f"Steps: {len(pipeline.steps)}")

            # Show steps
            if pipeline.steps:
                steps_df = pd.DataFrame([
                    {
                        'Step': i + 1,
                        'Name': step.name,
                        'Processor': step.processor_type,
                        'Function': step.function_name,
                        'Parameters': str(step.params),
                        'Dependencies': str(step.depends_on),
                        'Timestamp': step.timestamp
                    }
                    for i, step in enumerate(pipeline.steps)
                ])

                st.dataframe(steps_df, use_container_width=True)

                # Options to remove steps
                step_to_remove = st.number_input(
                    "Step to remove (0 to cancel):",
                    min_value=0,
                    max_value=len(pipeline.steps),
                    value=0
                )

                if step_to_remove > 0:
                    if st.button(f"Remove Step {step_to_remove}"):
                        pipeline.remove_step(step_to_remove - 1)
                        st.success(f"Removed step {step_to_remove}")
                        st.rerun()

            # Apply pipeline — the column-mapping preference is read OUTSIDE the
            # button so it persists across reruns.
            auto_map = st.checkbox(
                "Automatically map columns if needed", value=True, key="pipeline_auto_map"
            )

            if st.button("Apply Pipeline to Data"):
                try:
                    with st.spinner("Applying pipeline..."):
                        column_map = None if auto_map else {}
                        result_df = pipeline.apply(df, processors, column_map)
                    # Stage the result in session_state so the confirm step survives
                    # the rerun that clicking the confirm button triggers.
                    st.session_state["pipeline_apply_result"] = result_df
                except Exception as e:
                    st.session_state.pop("pipeline_apply_result", None)
                    st.error(f"Error applying pipeline: {str(e)}")

            # Preview + confirm are rendered OUTSIDE the apply-button block so that
            # "Confirm and Replace" is reachable on the following rerun.
            staged_result = st.session_state.get("pipeline_apply_result")
            if staged_result is not None:
                st.success("Pipeline applied successfully.")
                st.write("### Preview of Transformed Data")
                st.dataframe(staged_result.head(10), use_container_width=True)

                confirm_col, cancel_col = st.columns(2)
                with confirm_col:
                    if st.button("Confirm and Replace Current Data"):
                        st.session_state.pop("pipeline_apply_result", None)
                        return staged_result
                with cancel_col:
                    if st.button("Cancel"):
                        st.session_state.pop("pipeline_apply_result", None)
                        st.rerun()

    with step_config_tab:
        # Edit step configuration
        if st.session_state.current_pipeline in st.session_state.pipelines:
            pipeline = st.session_state.pipelines[st.session_state.current_pipeline]

            if not pipeline.steps:
                st.info("No steps in the current pipeline to edit.")
            else:
                st.subheader("Edit Step Configuration")

                step_to_edit = st.selectbox(
                    "Select step to edit:",
                    options=range(1, len(pipeline.steps) + 1),
                    format_func=lambda x: f"Step {x}: {pipeline.steps[x - 1].name}"
                )

                if step_to_edit:
                    step_index = step_to_edit - 1
                    step = pipeline.steps[step_index]

                    st.write(f"Editing step: {step.name}")
                    st.write(f"Processor type: {step.processor_type}")
                    st.write(f"Function: {step.function_name}")

                    # Create a custom name for the step
                    new_name = st.text_input("Step name:", value=step.name)

                    # Create input fields for each parameter
                    edited_params = {}

                    for param_name, param_value in step.params.items():
                        if param_name in ['column', 'column1', 'column2', 'x_column', 'y_column',
                                          'group_col', 'agg_col', 'date_col', 'value_col']:
                            # For column parameters, show a dropdown of available columns
                            edited_params[param_name] = st.selectbox(
                                f"{param_name}:",
                                options=df.columns.tolist(),
                                index=df.columns.tolist().index(param_value) if param_value in df.columns else 0,
                                key=f"edit_{step_index}_{param_name}"
                            )
                        elif isinstance(param_value, (int, float)):
                            edited_params[param_name] = st.number_input(
                                f"{param_name}:",
                                value=float(param_value),
                                key=f"edit_{step_index}_{param_name}"
                            )
                        elif isinstance(param_value, bool):
                            edited_params[param_name] = st.checkbox(
                                f"{param_name}:",
                                value=param_value,
                                key=f"edit_{step_index}_{param_name}"
                            )
                        elif isinstance(param_value, list):
                            if all(isinstance(x, (int, float)) for x in param_value):
                                # Numeric list - create number range options
                                options = list(range(1, 101))
                                edited_params[param_name] = st.multiselect(
                                    f"{param_name}:",
                                    options=options,
                                    default=[x for x in param_value if x in options],
                                    key=f"edit_{step_index}_{param_name}"
                                )
                            else:
                                # String list - likely columns
                                edited_params[param_name] = st.multiselect(
                                    f"{param_name}:",
                                    options=df.columns.tolist(),
                                    default=[x for x in param_value if x in df.columns],
                                    key=f"edit_{step_index}_{param_name}"
                                )
                        else:
                            # String or other values
                            edited_params[param_name] = st.text_input(
                                f"{param_name}:",
                                value=str(param_value),
                                key=f"edit_{step_index}_{param_name}"
                            )

                    # Dependencies management
                    st.subheader("Step Dependencies")
                    st.write("Set steps that must be executed before this step:")

                    # Show only steps that come before this one
                    available_deps = list(range(step_index))
                    if available_deps:
                        format_func = lambda x: f"Step {x + 1}: {pipeline.steps[x].name}"
                        dependencies = st.multiselect(
                            "This step depends on:",
                            options=available_deps,
                            default=step.depends_on,
                            format_func=format_func,
                            key=f"deps_{step_index}"
                        )
                    else:
                        dependencies = []
                        st.info("No prior steps available for dependencies.")

                    # Update step button
                    if st.button("Update Step Configuration"):
                        # Update step attributes
                        step.name = new_name
                        step.params = edited_params
                        step.depends_on = dependencies
                        st.success(f"Updated configuration for step: {step.name}")
                        st.rerun()

                    # Preview the step's effect
                    if st.button("Preview this step"):
                        try:
                            preview_df = step.apply(df, processors)
                            st.success(f"Step preview successful")

                            # Show differences
                            if set(preview_df.columns) != set(df.columns):
                                new_cols = [col for col in preview_df.columns if col not in df.columns]
                                if new_cols:
                                    st.write(f"New columns added: {', '.join(new_cols)}")
                                    st.dataframe(preview_df[new_cols].head(10), use_container_width=True)

                                removed_cols = [col for col in df.columns if col not in preview_df.columns]
                                if removed_cols:
                                    st.write(f"Columns removed: {', '.join(removed_cols)}")
                            else:
                                st.write("Same columns, but values may have changed")
                                st.dataframe(preview_df.head(10), use_container_width=True)

                        except Exception as e:
                            st.error(f"Error previewing step: {str(e)}")

    with export_tab:
        # Export pipeline to JSON
        if st.session_state.pipelines:
            pipeline_to_export = st.selectbox(
                "Select Pipeline to Export:",
                list(st.session_state.pipelines.keys()),
                key="export_pipeline_select"
            )

            if pipeline_to_export:
                pipeline = st.session_state.pipelines[pipeline_to_export]
                json_str = pipeline.to_json()

                # Create download link
                b64 = base64.b64encode(json_str.encode()).decode()
                filename = f"{pipeline.name.lower().replace(' ', '_')}_pipeline.json"
                href = f'<a href="data:file/json;base64,{b64}" download="{filename}">Download Pipeline JSON</a>'
                st.markdown(href, unsafe_allow_html=True)

                # Show JSON
                with st.expander("View Pipeline JSON"):
                    st.code(json_str, language="json")

                # Copy to clipboard button
                if st.button("Copy JSON to Clipboard"):
                    st.code(json_str, language="json")
                    st.success("JSON copied to clipboard (select and copy the code above)")
        else:
            st.info("No pipelines available for export.")

    with import_tab:
        # Import pipeline from JSON
        json_input = st.text_area("Paste Pipeline JSON:", height=200)

        if json_input:
            try:
                imported_pipeline = PreprocessingPipeline.from_json(json_input)
                st.success(f"Successfully parsed pipeline: {imported_pipeline.name}")

                # Option to rename
                new_name = st.text_input("Pipeline Name (leave empty to keep original):", "")

                if new_name:
                    imported_pipeline.name = new_name

                if st.button("Import Pipeline"):
                    st.session_state.pipelines[imported_pipeline.name] = imported_pipeline
                    st.session_state.current_pipeline = imported_pipeline.name
                    st.success(f"Imported pipeline: {imported_pipeline.name}")
                    st.rerun()

            except Exception as e:
                st.error(f"Error importing pipeline: {str(e)}")

        # Upload JSON file
        uploaded_file = st.file_uploader("Or upload pipeline JSON file:", type=["json"])

        if uploaded_file:
            try:
                json_str = uploaded_file.read().decode("utf-8")
                imported_pipeline = PreprocessingPipeline.from_json(json_str)
                st.success(f"Successfully parsed pipeline from file: {imported_pipeline.name}")

                # Option to rename
                new_name = st.text_input("Pipeline Name (file) (leave empty to keep original):", "")

                if new_name:
                    imported_pipeline.name = new_name

                if st.button("Import Pipeline from File"):
                    st.session_state.pipelines[imported_pipeline.name] = imported_pipeline
                    st.session_state.current_pipeline = imported_pipeline.name
                    st.success(f"Imported pipeline: {imported_pipeline.name}")
                    st.rerun()

            except Exception as e:
                st.error(f"Error importing pipeline from file: {str(e)}")

    # No transformed DataFrame to return unless the user confirmed above.
    return None


def record_preprocessing_step(processor_type, function_name, params, description=None):
    print(f"Recording step: {function_name} with params {params}")
    """
    Record a preprocessing step to the current pipeline

    Args:
        processor_type: Type of processor (e.g., 'numeric', 'categorical')
        function_name: Name of the function called
        params: Dictionary of parameters
        description: Optional description
    """
    # Check if pipeline management is initialized
    if 'pipelines' not in st.session_state:
        st.session_state.pipelines = {}

    if 'current_pipeline' not in st.session_state or st.session_state.current_pipeline is None:
        # Create a default pipeline if none exists
        default_name = "Default Pipeline"
        st.session_state.pipelines[default_name] = PreprocessingPipeline(default_name)
        st.session_state.current_pipeline = default_name

    # Get the current pipeline
    pipeline = st.session_state.pipelines[st.session_state.current_pipeline]

    # Create a step name if not provided
    step_name = description or f"{function_name} on {params.get('column', 'multiple columns')}"

    # Create and add the step
    step = PreprocessingStep(
        name=step_name,
        processor_type=processor_type,
        function_name=function_name,
        params=params,
        description=description
    )

    pipeline.add_step(step)

    # Notify user
    st.success(f"Added step to pipeline: {step_name}")