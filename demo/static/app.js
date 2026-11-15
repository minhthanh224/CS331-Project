(() => {
  const $ = (id) => document.getElementById(id);

  const dropZone = $("dropZone");
  const fileInput = $("fileInput");
  const previewBar = $("previewBar");
  const previewThumb = $("previewThumb");
  const previewName = $("previewName");
  const previewInfo = $("previewInfo");
  const clearBtn = $("clearBtn");
  const samplesWrap = $("samplesWrap");
  const samplesGrid = $("samplesGrid");

  const camMethodGroup = $("camMethodGroup");
  const clsThreshold = $("clsThreshold");
  const clsThresholdVal = $("clsThresholdVal");
  const camThreshold = $("camThreshold");
  const camThresholdVal = $("camThresholdVal");
  const crfAlpha = $("crfAlpha");
  const crfAlphaVal = $("crfAlphaVal");
  const useTta = $("useTta");
  const useCrf = $("useCrf");
  const useSam = $("useSam");
  const samToggleLabel = $("samToggleLabel");
  const runBtn = $("runBtn");

  const statusPills = $("statusPills");
  const emptyState = $("emptyState");
  const loadingBanner = $("loadingBanner");
  const errorBanner = $("errorBanner");
  const resultsContainer = $("resultsContainer");

  const clsSubtitle = $("clsSubtitle");
  const timingBadges = $("timingBadges");
  const activePills = $("activePills");
  const allProbsList = $("allProbsList");

  const maskOpacity = $("maskOpacity");
  const maskOpacityVal = $("maskOpacityVal");
  const pureMaskMode = $("pureMaskMode");

  const perClassSelector = $("perClassSelector");
  const perClassButtons = $("perClassButtons");

  const imgOriginal = $("imgOriginal");
  const imgDims = $("imgDims");
  const imgCam = $("imgCam");
  const camCaption = $("camCaption");
  const camMethodTag = $("camMethodTag");

  const baseVanilla = $("baseVanilla");
  const maskVanilla = $("maskVanilla");
  const cardCrf = $("cardCrf");
  const baseCrf = $("baseCrf");
  const maskCrf = $("maskCrf");
  const cardSam = $("cardSam");
  const baseSam = $("baseSam");
  const maskSam = $("maskSam");
  const legendList = $("legendList");

  const lightbox = $("lightbox");
  const lightboxTitle = $("lightboxTitle");
  const lightboxStage = lightbox.querySelector(".lightbox-stage");
  const lightboxBase = $("lightboxBase");
  const lightboxImg = $("lightboxImg");
  const lightboxClose = $("lightboxClose");

  let selectedFile = null;
  let camMethod = "cam";
  let lastResult = null;
  let selectedCamKey = "__max__";
  let samAvailable = true;

  function formatBytes(bytes) {
    if (!bytes) return "";
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  }

  function showError(msg) {
    errorBanner.textContent = msg;
    errorBanner.classList.remove("hidden");
  }

  function clearError() {
    errorBanner.textContent = "";
    errorBanner.classList.add("hidden");
  }

  function setPreview(file) {
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      previewThumb.src = e.target.result;
    };
    reader.readAsDataURL(file);
    previewName.textContent = file.name || "pasted-image.png";
    previewInfo.textContent = formatBytes(file.size);
    previewBar.classList.remove("hidden");
    runBtn.disabled = false;
  }

  function resetImage() {
    selectedFile = null;
    fileInput.value = "";
    previewBar.classList.add("hidden");
    runBtn.disabled = true;
    clearError();
  }

  async function handleFileSelection(file, autoRun = true) {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      showError("Tệp đã chọn không phải là ảnh hợp lệ.");
      return;
    }
    clearError();
    setPreview(file);
    if (autoRun) {
      await runInference();
    }
  }

  async function runInference() {
    if (!selectedFile) return;
    clearError();
    emptyState.classList.add("hidden");
    loadingBanner.classList.remove("hidden");
    runBtn.disabled = true;

    const form = new FormData();
    form.append("image", selectedFile);
    form.append("cam_method", camMethod);
    form.append("cls_threshold", clsThreshold.value);
    form.append("cam_threshold", camThreshold.value);
    form.append("crf_alpha", crfAlpha.value);
    form.append("use_tta", useTta.checked ? "true" : "false");
    form.append("use_crf", useCrf.checked ? "true" : "false");
    form.append("use_sam", (samAvailable && useSam.checked) ? "true" : "false");

    try {
      const res = await fetch("/api/predict", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || `Lỗi server (${res.status})`);
      }
      lastResult = data;
      renderResults(data);
    } catch (err) {
      showError(err.message || "Không thể kết nối tới máy chủ.");
    } finally {
      loadingBanner.classList.add("hidden");
      runBtn.disabled = !selectedFile;
    }
  }

  function renderResults(data) {
    resultsContainer.classList.remove("hidden");

    const { width, height } = data.image_size;
    imgDims.textContent = `${width} × ${height}`;
    camMethodTag.textContent = data.cam_method === "layercam" ? "LayerCAM" : "CAM";

    // Subtitle & fallback info
    if (data.fallback_used) {
      clsSubtitle.textContent =
        `Không có lớp nào vượt ngưỡng ${clsThreshold.value}; tự động lấy lớp xác suất cao nhất: ${data.active_classes[0]}.`;
    } else {
      clsSubtitle.textContent = `Phát hiện ${data.active_classes.length} lớp vượt ngưỡng ${clsThreshold.value}.`;
    }

    // Timing chips
    const t = data.timings || {};
    const chips = [
      ["Cls", t.cls_ms],
      ["CAM", t.cam_ms],
      ["CRF", t.crf_ms],
      ["SAM", t.sam_ms],
      ["Tổng", t.total_ms],
    ].filter(([, v]) => v !== undefined);

    timingBadges.innerHTML = chips
      .map(([k, v]) => `<span class="timing-chip">${k}: ${v} ms</span>`)
      .join("");

    // Active classes pills
    const probMap = new Map(data.probs.map((p) => [p.name, p]));
    activePills.innerHTML = data.active_classes
      .map((name) => {
        const item = probMap.get(name);
        const pct = item ? (item.prob * 100).toFixed(1) : "0.0";
        const color = item ? item.color : "#38bdf8";
        return `
          <div class="cls-chip">
            <span class="cls-dot" style="background:${color}"></span>
            <span>${name}</span>
            <span class="cls-prob">${pct}%</span>
          </div>
        `;
      })
      .join("");

    // All 20 classes bars
    allProbsList.innerHTML = data.probs
      .map((p) => {
        const pct = (p.prob * 100).toFixed(1);
        const barColor = p.active ? p.color : "#475569";
        return `
          <div class="prob-row">
            <div class="prob-row-top">
              <span>${p.name}</span>
              <span>${pct}%</span>
            </div>
            <div class="prob-bar-track">
              <div class="prob-bar-fill" style="width:${pct}%; background:${barColor}"></div>
            </div>
          </div>
        `;
      })
      .join("");

    // Images
    imgOriginal.src = data.images.original;
    baseVanilla.src = data.images.original;
    maskVanilla.src = data.images.vanilla_mask;

    if (data.images.crf_mask) {
      cardCrf.classList.remove("hidden");
      baseCrf.src = data.images.original;
      maskCrf.src = data.images.crf_mask;
    } else {
      cardCrf.classList.add("hidden");
    }

    if (data.images.sam_mask) {
      cardSam.classList.remove("hidden");
      baseSam.src = data.images.original;
      maskSam.src = data.images.sam_mask;
    } else {
      cardSam.classList.add("hidden");
    }

    // Per-class CAM selector
    selectedCamKey = "__max__";
    renderCamImage();
    const classNames = Object.keys(data.images.per_class_cams || {});
    if (classNames.length > 0) {
      perClassSelector.classList.remove("hidden");
      const btnDefs = [{ key: "__max__", label: "Tổng hợp (Max)" }].concat(
        classNames.map((c) => ({ key: c, label: c }))
      );
      perClassButtons.innerHTML = btnDefs
        .map(
          (b) =>
            `<button type="button" class="cls-tab-btn ${b.key === selectedCamKey ? "active" : ""}" data-cam-key="${b.key}">${b.label}</button>`
        )
        .join("");
    } else {
      perClassSelector.classList.add("hidden");
    }

    // Legend
    const legendItems = data.legend || [];
    if (legendItems.length === 0) {
      legendList.innerHTML = `<span class="cap-meta">Không có vùng foreground nào vượt ngưỡng.</span>`;
    } else {
      legendList.innerHTML = legendItems
        .map(
          (item) => `
          <div class="legend-item">
            <span class="cls-dot" style="background:${item.color}"></span>
            <span>${item.name}</span>
            <span class="cap-meta">${item.pixel_percent}%</span>
          </div>
        `
        )
        .join("");
    }
  }

  function renderCamImage() {
    if (!lastResult) return;
    if (selectedCamKey === "__max__") {
      imgCam.src = lastResult.images.cam_overlay;
      camCaption.textContent = "CAM Overlay (Max)";
    } else {
      imgCam.src = lastResult.images.per_class_cams[selectedCamKey] || lastResult.images.cam_overlay;
      camCaption.textContent = `CAM Overlay (${selectedCamKey})`;
    }
  }

  function openLightbox(kind) {
    if (!lastResult) return;
    lightboxStage.classList.remove("is-mask");
    lightboxBase.classList.add("hidden");

    if (kind === "original") {
      lightboxTitle.textContent = "Ảnh gốc";
      lightboxImg.src = lastResult.images.original;
    } else if (kind === "cam") {
      lightboxTitle.textContent = camCaption.textContent;
      lightboxImg.src = imgCam.src;
    } else if (kind === "vanilla") {
      lightboxTitle.textContent = "Vanilla CAM Mask";
      lightboxStage.classList.add("is-mask");
      lightboxBase.classList.remove("hidden");
      lightboxBase.src = lastResult.images.original;
      lightboxImg.src = lastResult.images.vanilla_mask;
    } else if (kind === "crf" && lastResult.images.crf_mask) {
      lightboxTitle.textContent = "CAM + DenseCRF Mask";
      lightboxStage.classList.add("is-mask");
      lightboxBase.classList.remove("hidden");
      lightboxBase.src = lastResult.images.original;
      lightboxImg.src = lastResult.images.crf_mask;
    } else if (kind === "sam" && lastResult.images.sam_mask) {
      lightboxTitle.textContent = "CAM + SAM (ViT-B) Mask";
      lightboxStage.classList.add("is-mask");
      lightboxBase.classList.remove("hidden");
      lightboxBase.src = lastResult.images.original;
      lightboxImg.src = lastResult.images.sam_mask;
    } else {
      return;
    }

    lightbox.classList.remove("hidden");
  }

  // Event bindings: Drop zone & File input
  dropZone.addEventListener("click", () => fileInput.click());
  dropZone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files[0]) {
      handleFileSelection(fileInput.files[0], true);
    }
  });

  ["dragenter", "dragover"].forEach((evtName) => {
    dropZone.addEventListener(evtName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add("drag-over");
    });
  });

  ["dragleave", "drop"].forEach((evtName) => {
    dropZone.addEventListener(evtName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove("drag-over");
    });
  });

  dropZone.addEventListener("drop", (e) => {
    const files = e.dataTransfer && e.dataTransfer.files;
    if (files && files[0]) {
      handleFileSelection(files[0], true);
    }
  });

  // Prevent browser from opening dropped files outside the drop zone, and allow dropping anywhere
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => {
    e.preventDefault();
    if (e.target !== dropZone && !dropZone.contains(e.target)) {
      const files = e.dataTransfer && e.dataTransfer.files;
      if (files && files[0]) {
        handleFileSelection(files[0], true);
      }
    }
  });

  // Clipboard paste (Ctrl+V)
  document.addEventListener("paste", (e) => {
    const items = e.clipboardData && e.clipboardData.items;
    if (!items) return;
    for (const item of items) {
      if (item.type.startsWith("image/")) {
        const file = item.getAsFile();
        if (file) {
          handleFileSelection(file, true);
          break;
        }
      }
    }
  });

  clearBtn.addEventListener("click", resetImage);
  runBtn.addEventListener("click", runInference);

  // Controls
  camMethodGroup.addEventListener("click", (e) => {
    const btn = e.target.closest(".seg-btn");
    if (!btn) return;
    camMethod = btn.dataset.method;
    camMethodGroup.querySelectorAll(".seg-btn").forEach((b) => b.classList.toggle("active", b === btn));
  });

  clsThreshold.addEventListener("input", () => {
    clsThresholdVal.textContent = Number(clsThreshold.value).toFixed(2);
  });
  camThreshold.addEventListener("input", () => {
    camThresholdVal.textContent = Number(camThreshold.value).toFixed(2);
  });
  crfAlpha.addEventListener("input", () => {
    crfAlphaVal.textContent = Number(crfAlpha.value).toFixed(1);
  });

  maskOpacity.addEventListener("input", () => {
    const val = Number(maskOpacity.value);
    document.documentElement.style.setProperty("--mask-opacity", val.toString());
    maskOpacityVal.textContent = `${Math.round(val * 100)}%`;
  });

  pureMaskMode.addEventListener("change", () => {
    document.body.classList.toggle("pure-mask", pureMaskMode.checked);
  });

  perClassButtons.addEventListener("click", (e) => {
    const btn = e.target.closest(".cls-tab-btn");
    if (!btn) return;
    selectedCamKey = btn.dataset.camKey;
    perClassButtons.querySelectorAll(".cls-tab-btn").forEach((b) => b.classList.toggle("active", b === btn));
    renderCamImage();
  });

  document.querySelectorAll(".vis-card").forEach((card) => {
    card.addEventListener("click", () => openLightbox(card.dataset.zoom));
  });

  lightboxClose.addEventListener("click", () => lightbox.classList.add("hidden"));
  lightbox.querySelector(".lightbox-backdrop").addEventListener("click", () => lightbox.classList.add("hidden"));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") lightbox.classList.add("hidden");
  });

  // Init status & sample images
  async function init() {
    try {
      const res = await fetch("/api/status");
      const status = await res.json();
      if (!res.ok) throw new Error(status.error || "Status error");

      samAvailable = Boolean(status.sam_available);
      if (!samAvailable) {
        useSam.checked = false;
        useSam.disabled = true;
        samToggleLabel.classList.add("disabled");
        samToggleLabel.title = status.sam_error || "SAM không khả dụng";
      }

      statusPills.innerHTML = `
        <span class="pill pill-ok">ResNet-101 (${status.checkpoint})</span>
        <span class="pill pill-ok">Device: ${status.device.toUpperCase()}</span>
        <span class="pill ${samAvailable ? "pill-ok" : "pill-warn"}">
          SAM ViT-B: ${samAvailable ? "Sẵn sàng" : "Không khả dụng"}
        </span>
      `;
    } catch (err) {
      statusPills.innerHTML = `<span class="pill pill-warn">Lỗi khởi tạo: ${err.message}</span>`;
    }

    try {
      const res = await fetch("/api/samples");
      const payload = await res.json();
      const samples = payload.samples || [];
      if (samples.length > 0) {
        samplesWrap.classList.remove("hidden");
        samplesGrid.innerHTML = samples
          .map(
            (s) => `
            <button type="button" class="sample-thumb" data-url="${s.url}" data-name="${s.filename}" title="Thử ảnh ${s.filename}">
              <img src="${s.url}" alt="${s.filename}" loading="lazy" />
            </button>
          `
          )
          .join("");

        samplesGrid.addEventListener("click", async (e) => {
          const btn = e.target.closest(".sample-thumb");
          if (!btn) return;
          try {
            const imgRes = await fetch(btn.dataset.url);
            const blob = await imgRes.blob();
            const file = new File([blob], btn.dataset.name, { type: blob.type || "image/jpeg" });
            await handleFileSelection(file, true);
          } catch (err) {
            showError("Không thể tải ảnh mẫu: " + err.message);
          }
        });
      }
    } catch {
      // Optional samples bar; ignore if unavailable
    }
  }

  init();
})();

