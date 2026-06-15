import streamlit as st
import matplotlib.pyplot as plt
import io


def export_chart(fig, chart_name="chart"):
    """
    Export a Plotly chart as interactive HTML only
    No to_image functionality - using reliable HTML export only

    Args:
        fig: A Plotly figure object
        chart_name: Base name for the exported file
    """
    try:
        # Convert to HTML (reliable method that works)
        html_bytes = fig.to_html(include_plotlyjs='cdn', config={"displayModeBar": True}).encode()

        # Add screenshot capability to the HTML with button at bottom
        html_with_screenshot = _add_screenshot_capability(html_bytes.decode(), chart_name)

        # Provide download button for HTML
        st.download_button(
            label="📥 Download Interactive Chart",
            data=html_with_screenshot.encode(),
            file_name=f"{chart_name}.html",
            mime="text/html"
        )

    except Exception as e:
        st.error(f"Error exporting chart: {str(e)}")


def export_advanced_chart(fig, chart_name="advanced_chart"):
    """
    Export function for advanced Plotly charts - HTML only
    """
    export_chart(fig, chart_name)


def export_matplotlib_chart(fig_obj, chart_name="mpl_chart", dpi=100, width=None, height=None):
    """
    Export a Matplotlib figure as PNG

    Args:
        fig_obj: The Matplotlib figure object or plt module
        chart_name: Base name for the exported file
        dpi: Resolution of the image
        width, height: Figure dimensions in inches (optional)
    """
    try:
        # First determine what kind of object we received
        if fig_obj == plt:
            # If plt module was passed, get the current figure
            fig = plt.gcf()  # Get current figure
        elif hasattr(fig_obj, 'figure'):
            # If an Axes object was passed, get its figure
            fig = fig_obj.figure
        else:
            # Assume it's already a Figure object
            fig = fig_obj

        # Set figure size if specified
        if width is not None or height is not None:
            current_width, current_height = fig.get_size_inches()
            new_width = width if width is not None else current_width
            new_height = height if height is not None else current_height
            fig.set_size_inches(new_width, new_height)

        # Save figure to bytes
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)

        # Provide download button
        st.download_button(
            label=f"📥 Download Chart (PNG)",
            data=buf,
            file_name=f"{chart_name}.png",
            mime="image/png"
        )

        # Always close matplotlib figure to avoid memory issues
        plt.close(fig)

    except Exception as e:
        # Try to close the figure even on error
        try:
            if 'fig' in locals():
                plt.close(fig)
        except:
            pass
        st.error(f"Error exporting Matplotlib chart: {str(e)}")


def _add_screenshot_capability(html_content, chart_name):
    """
    Add JavaScript to enable taking screenshots within the HTML
    With format selection (PNG, JPEG, WebP)
    Button placed at bottom of screen, not covering the chart
    """
    # JavaScript for screenshot functionality with format selection
    screenshot_js = """
    <script src="https://html2canvas.hertzen.com/dist/html2canvas.min.js"></script>
    <style>
    .screenshot-button {
        position: fixed;
        bottom: 20px;
        right: 20px;
        padding: 8px 16px;
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        z-index: 9999;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .screenshot-button:hover {
        background-color: #45a049;
    }
    .format-panel {
        position: fixed;
        bottom: 70px;
        right: 20px;
        padding: 15px;
        background-color: white;
        border: 1px solid #ddd;
        border-radius: 4px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        z-index: 9999;
        display: none;
    }
    .format-panel h3 {
        margin-top: 0;
        margin-bottom: 10px;
        font-size: 16px;
    }
    .format-option {
        display: flex;
        align-items: center;
        margin-bottom: 8px;
    }
    .format-option input {
        margin-right: 8px;
    }
    .format-option label {
        font-size: 14px;
    }
    .format-buttons {
        display: flex;
        justify-content: space-between;
        margin-top: 10px;
    }
    .format-button {
        padding: 5px 10px;
        border: none;
        border-radius: 3px;
        cursor: pointer;
        font-size: 14px;
    }
    .download-button {
        background-color: #4CAF50;
        color: white;
    }
    .download-button:hover {
        background-color: #45a049;
    }
    .cancel-button {
        background-color: #f1f1f1;
    }
    .cancel-button:hover {
        background-color: #e0e0e0;
    }
    .quality-slider {
        width: 100%;
        margin-top: 5px;
    }
    .quality-value {
        text-align: center;
        font-size: 12px;
        margin-top: 2px;
    }
    </style>

    <button id="screenshot-button" class="screenshot-button" onclick="toggleFormatPanel()">📷 Save as Image</button>

    <div id="format-panel" class="format-panel">
        <h3>Save Chart As:</h3>

        <div class="format-option">
            <input type="radio" id="format-png" name="format" value="png" checked>
            <label for="format-png">PNG (Highest Quality)</label>
        </div>

        <div class="format-option">
            <input type="radio" id="format-jpeg" name="format" value="jpeg">
            <label for="format-jpeg">JPEG (Smaller File Size)</label>
        </div>

        <div class="format-option">
            <input type="radio" id="format-webp" name="format" value="webp">
            <label for="format-webp">WebP (Modern Format)</label>
        </div>

        <div id="quality-container" style="display: none; margin-top: 10px;">
            <label for="quality-slider">Quality: </label>
            <input type="range" id="quality-slider" class="quality-slider" min="0" max="100" value="90">
            <div id="quality-value" class="quality-value">90%</div>
        </div>

        <div class="format-buttons">
            <button class="format-button cancel-button" onclick="toggleFormatPanel()">Cancel</button>
            <button class="format-button download-button" onclick="takeScreenshot()">Download</button>
        </div>
    </div>

    <script>
    // Toggle the format panel visibility
    function toggleFormatPanel() {
        const panel = document.getElementById('format-panel');
        panel.style.display = panel.style.display === 'none' || panel.style.display === '' ? 'block' : 'none';
    }

    // Show quality slider for JPEG and WebP
    document.querySelectorAll('input[name="format"]').forEach(input => {
        input.addEventListener('change', function() {
            const qualityContainer = document.getElementById('quality-container');
            if (this.value === 'jpeg' || this.value === 'webp') {
                qualityContainer.style.display = 'block';
            } else {
                qualityContainer.style.display = 'none';
            }
        });
    });

    // Update quality value display
    document.getElementById('quality-slider').addEventListener('input', function() {
        document.getElementById('quality-value').textContent = this.value + '%';
    });

    // Take screenshot and download
    function takeScreenshot() {
        // Find the plot element
        const plotElement = document.querySelector('.plotly-graph-div');
        if (!plotElement) {
            alert('Plot element not found!');
            return;
        }

        // Get selected format
        const formatInput = document.querySelector('input[name="format"]:checked');
        const format = formatInput ? formatInput.value : 'png';

        // Get quality for JPEG/WebP
        let quality = 1.0;
        if (format === 'jpeg' || format === 'webp') {
            quality = parseInt(document.getElementById('quality-slider').value) / 100;
        }

        // Take screenshot using html2canvas
        html2canvas(plotElement, {
            scale: 2, // Higher scale for better quality
            backgroundColor: null
        }).then(canvas => {
            // Create download link
            const link = document.createElement('a');
            link.download = '""" + chart_name + """.' + format;

            // Convert canvas to data URL with selected format
            if (format === 'png') {
                link.href = canvas.toDataURL('image/png');
            } else if (format === 'jpeg') {
                link.href = canvas.toDataURL('image/jpeg', quality);
            } else if (format === 'webp') {
                link.href = canvas.toDataURL('image/webp', quality);
            }

            // Trigger download
            link.click();

            // Hide format panel
            toggleFormatPanel();
        });
    }
    </script>
    """

    # Insert the script just before the closing body tag
    if "</body>" in html_content:
        return html_content.replace("</body>", screenshot_js + "</body>")
    else:
        return html_content + screenshot_js