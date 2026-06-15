import streamlit as st
import streamlit.components.v1 as components
import os
from config import Config, Paths

# Configure page
st.set_page_config(
    layout="wide",
    initial_sidebar_state="collapsed",
    page_title=Config.APP_NAME
)

# Apply the global CSS from config
st.markdown(Config.get_css(), unsafe_allow_html=True)

# ── Hero: interactive WebGL "data network" (self-contained three.js in a
#    sandboxed components iframe; CDN three.js, no new Python dependency) ──
_HERO_HTML = r"""
<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Space+Grotesk:wght@600;700&display=swap" rel="stylesheet">
<style>
  html,body{margin:0;height:100%;background:transparent;overflow:hidden}
  #wrap{position:relative;width:100%;height:100%}
  #c{display:block;width:100%;height:100%}
  #overlay{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;
    justify-content:center;text-align:center;pointer-events:none;padding:0 18px}
  #overlay h1{margin:0 0 .55rem;font-family:'Space Grotesk','Inter',system-ui,sans-serif;
    font-size:clamp(2.2rem,5.2vw,3.7rem);font-weight:700;letter-spacing:-.02em;line-height:1.04;
    background:linear-gradient(135deg,#a5b4fc 0%,#c4b5fd 48%,#93c5fd 100%);
    -webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;
    filter:drop-shadow(0 4px 34px rgba(99,102,241,.35))}
  #overlay p{margin:0;max-width:660px;color:#cbd5e1;font-family:'Inter',system-ui,sans-serif;
    font-size:clamp(.95rem,2vw,1.14rem);line-height:1.6}
  #fade{position:absolute;left:0;right:0;bottom:0;height:90px;pointer-events:none;
    background:linear-gradient(to bottom,rgba(11,16,32,0),rgba(11,16,32,1))}
</style></head>
<body><div id="wrap">
  <canvas id="c"></canvas>
  <div id="overlay"><h1>__APP_NAME__</h1><p>__TAGLINE__</p></div>
  <div id="fade"></div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
(function(){
  if(!window.THREE){return;}
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var canvas=document.getElementById('c');
  var renderer=new THREE.WebGLRenderer({canvas:canvas,alpha:true,antialias:true});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,2));
  var scene=new THREE.Scene();
  var camera=new THREE.PerspectiveCamera(60,1,1,3000); camera.position.z=440;
  function size(){var W=canvas.clientWidth,H=canvas.clientHeight;renderer.setSize(W,H,false);
    camera.aspect=W/(H||1);camera.updateProjectionMatrix();}
  var N=130, SP={x:700,y:235,z:150}, LINK=165, LINK2=LINK*LINK;
  var pos=new Float32Array(N*3), vel=[];
  for(var i=0;i<N;i++){
    pos[i*3]=(Math.random()*2-1)*SP.x; pos[i*3+1]=(Math.random()*2-1)*SP.y; pos[i*3+2]=(Math.random()*2-1)*SP.z;
    vel.push({x:(Math.random()*2-1)*0.22,y:(Math.random()*2-1)*0.22,z:(Math.random()*2-1)*0.16});
  }
  var pGeo=new THREE.BufferGeometry(); pGeo.setAttribute('position',new THREE.BufferAttribute(pos,3));
  function dotTex(){var s=64,cv=document.createElement('canvas');cv.width=cv.height=s;var g=cv.getContext('2d');
    var gr=g.createRadialGradient(s/2,s/2,0,s/2,s/2,s/2);
    gr.addColorStop(0,'rgba(224,231,255,1)');gr.addColorStop(.35,'rgba(129,140,248,.95)');gr.addColorStop(1,'rgba(99,102,241,0)');
    g.fillStyle=gr;g.beginPath();g.arc(s/2,s/2,s/2,0,Math.PI*2);g.fill();return new THREE.CanvasTexture(cv);}
  var pMat=new THREE.PointsMaterial({size:13,map:dotTex(),transparent:true,depthWrite:false,blending:THREE.AdditiveBlending});
  scene.add(new THREE.Points(pGeo,pMat));
  var maxL=N*9, lpos=new Float32Array(maxL*6);
  var lGeo=new THREE.BufferGeometry(); lGeo.setAttribute('position',new THREE.BufferAttribute(lpos,3));
  var lMat=new THREE.LineBasicMaterial({color:0x8b9bf6,transparent:true,opacity:0.42,blending:THREE.AdditiveBlending});
  scene.add(new THREE.LineSegments(lGeo,lMat));
  var mouse={x:0,y:0,on:false};
  canvas.addEventListener('pointermove',function(e){var r=canvas.getBoundingClientRect();
    mouse.x=((e.clientX-r.left)/r.width)*2-1; mouse.y=-(((e.clientY-r.top)/r.height)*2-1); mouse.on=true;});
  canvas.addEventListener('pointerleave',function(){mouse.on=false;});
  function step(){
    var mx=mouse.x*SP.x*1.05, my=mouse.y*SP.y*1.4;
    for(var i=0;i<N;i++){
      var x=pos[i*3]+vel[i].x, y=pos[i*3+1]+vel[i].y, z=pos[i*3+2]+vel[i].z;
      if(x>SP.x||x<-SP.x)vel[i].x*=-1; if(y>SP.y||y<-SP.y)vel[i].y*=-1; if(z>SP.z||z<-SP.z)vel[i].z*=-1;
      if(mouse.on){var dx=x-mx,dy=y-my,d2=dx*dx+dy*dy; if(d2<15000&&d2>0.01){var d=Math.sqrt(d2),f=(15000-d2)/15000*1.7; x+=dx/d*f; y+=dy/d*f;}}
      pos[i*3]=x;pos[i*3+1]=y;pos[i*3+2]=z;
    }
    pGeo.attributes.position.needsUpdate=true;
    var n=0;
    for(var a=0;a<N;a++){for(var b=a+1;b<N;b++){
      var ddx=pos[a*3]-pos[b*3],ddy=pos[a*3+1]-pos[b*3+1],ddz=pos[a*3+2]-pos[b*3+2];
      if(ddx*ddx+ddy*ddy+ddz*ddz<LINK2 && n<maxL){
        lpos[n*6]=pos[a*3];lpos[n*6+1]=pos[a*3+1];lpos[n*6+2]=pos[a*3+2];
        lpos[n*6+3]=pos[b*3];lpos[n*6+4]=pos[b*3+1];lpos[n*6+5]=pos[b*3+2];n++;}
    }}
    lGeo.setDrawRange(0,n*2); lGeo.attributes.position.needsUpdate=true;
    camera.position.x+=((mouse.x*44)-camera.position.x)*0.04;
    camera.position.y+=((mouse.y*26)-camera.position.y)*0.04; camera.lookAt(0,0,0);
  }
  var req;
  function loop(){step();renderer.render(scene,camera);req=requestAnimationFrame(loop);}
  function start(){size(); if(reduce){step();renderer.render(scene,camera);} else {cancelAnimationFrame(req);loop();}}
  window.addEventListener('resize',size);
  start(); setTimeout(size,60);
})();
</script></body></html>
"""
_hero = (_HERO_HTML
         .replace("__APP_NAME__", Config.APP_NAME)
         .replace("__TAGLINE__", "Upload a spreadsheet and understand your data — clean it, chart it, "
                                 "and build prediction models, no code."))
components.html(_hero, height=420, scrolling=False)

# Main content in two columns
left, right = st.columns([1, 1])

with left:
    st.markdown('<div class="content-area">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: white;">From spreadsheet to insights — no code</h3>', unsafe_allow_html=True)
    st.markdown('<p>Everything you need, step by step:</p>', unsafe_allow_html=True)

    # Checkboxes with green check marks
    st.markdown("""
    <div class="checkbox-item">
        <span class="green-check">✅</span> Spot data-quality issues automatically
    </div>
    <div class="checkbox-item">
        <span class="green-check">✅</span> Clean &amp; prepare your data — no formulas
    </div>
    <div class="checkbox-item">
        <span class="green-check">✅</span> Explore with interactive charts &amp; dashboards
    </div>
    <div class="checkbox-item">
        <span class="green-check">✅</span> Build &amp; download prediction models
    </div>
    <p style="margin-top: 1rem;">Have a spreadsheet? Get started — or try it with sample data.</p>
    """, unsafe_allow_html=True)

    # Button
    if st.button('Get Started'):
        st.switch_page("pages/1_upload_data.py")

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    st.image(os.path.join(Paths.STATIC_DIR, 'laptop.PNG'))
    st.markdown('</div>', unsafe_allow_html=True)

# Add a simple footer
st.markdown("""
<div style="text-align: center; margin-top: 1rem; color: #6B7280; font-size: 0.8rem;">
    © 2025 Data Analysis Platform • All rights reserved
</div>
""", unsafe_allow_html=True)