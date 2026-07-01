"""A little CSS to make the app look like a real product, not a bare demo."""

CSS = """
<style>
:root {
  --nova: #5b6cff;
  --nova-soft: #eef0ff;
  --ink: #1f2330;
  --muted: #6b7280;
}
.block-container { padding-top: 2rem; max-width: 1100px; }
.nova-header {
  display:flex; align-items:center; gap:.7rem; margin-bottom:.2rem;
}
.nova-header .logo {
  width:38px; height:38px; border-radius:10px; background:linear-gradient(135deg,#5b6cff,#9b5bff);
  display:flex; align-items:center; justify-content:center; color:#fff; font-weight:700;
}
.nova-title { font-size:1.45rem; font-weight:700; color:var(--ink); line-height:1.1; }
.nova-sub { color:var(--muted); font-size:.85rem; }

.card {
  border:1px solid #e7e8ee; border-radius:14px; padding:14px 16px; background:#fff;
  box-shadow:0 1px 2px rgba(20,20,40,.04);
}
.profile-row { display:flex; justify-content:space-between; font-size:.86rem; padding:3px 0; }
.profile-row span:first-child { color:var(--muted); }
.profile-row span:last-child { font-weight:600; color:var(--ink); }

.badge { display:inline-block; padding:2px 10px; border-radius:999px; font-size:.74rem;
         font-weight:600; margin-right:6px; }
.b-billing   { background:#e6f4ff; color:#0a66c2; }
.b-technical { background:#eafaf0; color:#0a7d3a; }
.b-account   { background:#fdf0e6; color:#b4530a; }
.b-general,.b-direct { background:#eef0ff; color:#4a4fbf; }
.b-low{background:#eef2f6;color:#52606d}.b-normal{background:#e6f4ff;color:#0a66c2}
.b-high{background:#fff3e0;color:#b4530a}.b-urgent{background:#fde8e8;color:#c81e1e}
.b-happy{background:#eafaf0;color:#0a7d3a}.b-neutral{background:#eef2f6;color:#52606d}
.b-frustrated{background:#fff3e0;color:#b4530a}.b-angry{background:#fde8e8;color:#c81e1e}

.memory-pill { display:block; background:var(--nova-soft); color:#3a3f8f; border-radius:8px;
  padding:6px 10px; font-size:.82rem; margin-bottom:6px; }
.trace-line { font-family:ui-monospace,Menlo,monospace; font-size:.78rem; color:#444;
  padding:2px 0; }
.approve-card { border:2px solid #f0b429; background:#fffaf0; border-radius:14px; padding:16px; }

/* ── Agent Activity timeline ─────────────────────────────────────────────── */
.tl { border-left:2px solid rgba(255,255,255,.2); margin:4px 0 4px 14px; padding-left:0; }
.tl-empty { color:rgba(255,255,255,.55); font-size:.85rem; padding:14px 6px; }
.tl-title-bar { font-size:.78rem; color:rgba(255,255,255,.6); margin:0 0 10px 14px;
  font-style:italic; }
.tl-item { position:relative; padding:0 0 14px 22px; }
.tl-dot { position:absolute; left:-13px; top:-2px; width:24px; height:24px;
  background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.25); border-radius:50%;
  display:flex; align-items:center; justify-content:center; font-size:.8rem; }
.tl-agent { font-size:.68rem; font-weight:700; letter-spacing:.04em;
  text-transform:uppercase; color:rgba(255,255,255,.6); }
.tl-head { font-size:.86rem; font-weight:600; color:#ffffff; margin-top:1px; }
.tl-detail { font-family:ui-monospace,Menlo,monospace; font-size:.76rem; color:rgba(255,255,255,.8);
  background:rgba(255,255,255,.08); border-radius:7px; padding:5px 8px; margin-top:4px;
  white-space:pre-wrap; word-break:break-word; }
</style>
"""
