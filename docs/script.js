/* HOPE explainer — interactions */
(function () {
  "use strict";
  var root = document.documentElement;

  /* ---- persisted preferences ---- */
  var savedLang = localStorage.getItem("hope-lang");
  var savedTheme = localStorage.getItem("hope-theme");
  root.setAttribute("data-lang", savedLang || "ko");
  if (savedTheme) {
    root.setAttribute("data-theme", savedTheme);
  } else if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
    root.setAttribute("data-theme", "dark");
  } else {
    root.setAttribute("data-theme", "light");
  }

  function ready(fn) {
    if (document.readyState !== "loading") fn();
    else document.addEventListener("DOMContentLoaded", fn);
  }

  ready(function () {
    /* ---- language toggle ---- */
    var langBtns = document.querySelectorAll(".langtoggle button");
    function syncLang() {
      var l = root.getAttribute("data-lang");
      langBtns.forEach(function (b) { b.classList.toggle("active", b.dataset.lang === l); });
      document.documentElement.lang = l;
    }
    langBtns.forEach(function (b) {
      b.addEventListener("click", function () {
        root.setAttribute("data-lang", b.dataset.lang);
        localStorage.setItem("hope-lang", b.dataset.lang);
        syncLang();
      });
    });
    syncLang();

    /* ---- theme toggle ---- */
    var themeBtn = document.getElementById("themeBtn");
    function syncTheme() {
      var t = root.getAttribute("data-theme");
      if (themeBtn) themeBtn.textContent = t === "dark" ? "◑ Light" : "◐ Dark";
    }
    if (themeBtn) {
      themeBtn.addEventListener("click", function () {
        var t = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
        root.setAttribute("data-theme", t);
        localStorage.setItem("hope-theme", t);
        syncTheme();
      });
    }
    syncTheme();

    /* ---- reading progress ---- */
    var bar = document.getElementById("progress");
    function onScroll() {
      var h = document.documentElement;
      var max = h.scrollHeight - h.clientHeight;
      var p = max > 0 ? (h.scrollTop || document.body.scrollTop) / max : 0;
      if (bar) bar.style.width = (p * 100).toFixed(2) + "%";
    }
    document.addEventListener("scroll", onScroll, { passive: true });
    onScroll();

    /* ---- scrollspy TOC ---- */
    var tocLinks = Array.prototype.slice.call(document.querySelectorAll(".toc-sticky a"));
    var sections = tocLinks
      .map(function (a) { return document.getElementById(a.getAttribute("href").slice(1)); })
      .filter(Boolean);
    var spy = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) {
          var id = e.target.id;
          tocLinks.forEach(function (a) {
            a.classList.toggle("active", a.getAttribute("href") === "#" + id);
          });
        }
      });
    }, { rootMargin: "-15% 0px -70% 0px", threshold: 0 });
    sections.forEach(function (s) { spy.observe(s); });

    /* ---- reveal on scroll ---- */
    var reveals = document.querySelectorAll(".reveal");
    var rObs = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) { e.target.classList.add("in"); rObs.unobserve(e.target); }
      });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.06 });
    reveals.forEach(function (r) { rObs.observe(r); });

    /* ---- render KaTeX ---- */
    if (window.renderMathInElement) {
      window.renderMathInElement(document.body, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "\\[", right: "\\]", display: true },
          { left: "$", right: "$", display: false },
          { left: "\\(", right: "\\)", display: false }
        ],
        throwOnError: false
      });
    }

    /* ---- highlight code ---- */
    if (window.hljs) {
      document.querySelectorAll("pre code").forEach(function (b) { window.hljs.highlightElement(b); });
    }

    drawBands();
  });

  /* ---- hero oscillation canvas: gamma / beta / delta brainwaves ---- */
  var canvas = document.getElementById("osc");
  if (canvas && canvas.getContext) {
    var ctx = canvas.getContext("2d");
    var w, h, dpr;
    function resize() {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = canvas.clientWidth; h = canvas.clientHeight;
      canvas.width = w * dpr; canvas.height = h * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    window.addEventListener("resize", resize);
    resize();

    function accent(name) {
      return getComputedStyle(root).getPropertyValue(name).trim() || "#b4541e";
    }
    var reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    var t0 = 0;
    // three frequency bands: fast (gamma) → slow (delta), mirroring CMS update rates
    var waves = [
      { freq: 0.018, amp: 0.10, speed: 0.9, varname: "--accent-3", yfrac: 0.34, lw: 1.2 },
      { freq: 0.010, amp: 0.15, speed: 0.55, varname: "--accent-2", yfrac: 0.5, lw: 1.6 },
      { freq: 0.0045, amp: 0.22, speed: 0.28, varname: "--accent", yfrac: 0.66, lw: 2.2 }
    ];
    function frame(ts) {
      if (!t0) t0 = ts;
      var time = (ts - t0) / 1000;
      ctx.clearRect(0, 0, w, h);
      waves.forEach(function (wv) {
        ctx.beginPath();
        ctx.lineWidth = wv.lw;
        ctx.strokeStyle = accent(wv.varname);
        ctx.globalAlpha = 0.5;
        var baseY = h * wv.yfrac;
        for (var x = 0; x <= w; x += 4) {
          var env = Math.sin((x / w) * Math.PI); // fade at edges
          var y = baseY + Math.sin(x * wv.freq + time * wv.speed * (reduced ? 0 : 1)) * (h * wv.amp) * env;
          if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
      });
      ctx.globalAlpha = 1;
      if (!reduced) requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }

  /* ---- mini band sparkline waves in the CMS section ---- */
  function drawBands() {
    document.querySelectorAll("canvas.wave").forEach(function (cv) {
      var c = cv.getContext("2d"); if (!c) return;
      var dpr = Math.min(window.devicePixelRatio || 1, 2);
      var W = cv.clientWidth || 200, H = cv.clientHeight || 38;
      cv.width = W * dpr; cv.height = H * dpr; c.setTransform(dpr,0,0,dpr,0,0);
      var freq = parseFloat(cv.dataset.freq || "0.05");
      var col = getComputedStyle(root).getPropertyValue(cv.dataset.col || "--accent").trim() || "#b4541e";
      c.clearRect(0,0,W,H); c.beginPath(); c.lineWidth = 1.6; c.strokeStyle = col;
      for (var x=0;x<=W;x+=2){ var y=H/2 + Math.sin(x*freq)*(H*0.32); if(x===0)c.moveTo(x,y); else c.lineTo(x,y); }
      c.stroke();
    });
  }
  window.addEventListener("resize", function(){ setTimeout(drawBands, 120); });
})();
