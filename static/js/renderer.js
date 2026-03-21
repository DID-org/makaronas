/* ==========================================================================
   Makaronas — renderer.js
   PresentationBlock renderers: dispatch by type, markdown pipeline,
   asset URL resolution. All block rendering goes through window.Renderer.
   ========================================================================== */

(function () {
  'use strict';

  // --------------------------------------------------------------------------
  // Markdown Pipeline — marked.js + DOMPurify
  // --------------------------------------------------------------------------

  // Configure marked: override link renderer to add target="_blank"
  var markedRenderer = new marked.Renderer();
  markedRenderer.link = function (token) {
    var href = token.href || '';
    var text = token.text || '';
    return '<a href="' + href + '" target="_blank" rel="noopener noreferrer">' + text + '</a>';
  };

  marked.setOptions({
    renderer: markedRenderer,
    breaks: false,
    gfm: true
  });

  // DOMPurify config: strict whitelist, allow target and rel on links.
  // No <img> from markdown — images come from ImageBlock.
  // No headings or blockquote — start strict, expand later if needed.
  var PURIFY_CONFIG = {
    ALLOWED_TAGS: ['p', 'strong', 'em', 'a', 'ul', 'ol', 'li', 'br'],
    ALLOWED_ATTR: ['href', 'target', 'rel'],
    ADD_ATTR: ['target', 'rel']
  };

  /**
   * Render markdown text to sanitized HTML string.
   * Shared pipeline used by block renderers, dialogue (5b), and streaming (6a).
   */
  function renderMarkdown(text) {
    if (!text) return '';
    var rawHtml = marked.parse(text);
    return DOMPurify.sanitize(rawHtml, PURIFY_CONFIG);
  }

  // --------------------------------------------------------------------------
  // Asset Error Fallback Helper
  // --------------------------------------------------------------------------

  /**
   * Replace a broken media element with a text fallback.
   * Used by image, audio, and video renderers on error events.
   */
  function handleAssetError(mediaEl, fallbackText) {
    mediaEl.style.display = 'none';
    var fallback = document.createElement('div');
    fallback.className = 'block-asset-error';
    fallback.textContent = fallbackText || window.I18n.error_asset;
    mediaEl.parentNode.insertBefore(fallback, mediaEl.nextSibling);
  }

  // --------------------------------------------------------------------------
  // Block Renderers
  // --------------------------------------------------------------------------

  function renderTextBlock(block) {
    var el = document.createElement('div');
    el.className = 'block block-text';
    if (block.style) {
      el.className += ' block-text--' + block.style;
    }
    el.id = 'block-' + block.id;
    el.innerHTML = renderMarkdown(block.text);
    return el;
  }

  function renderImageBlock(block, taskId) {
    var figure = document.createElement('figure');
    figure.className = 'block block-image';
    figure.id = 'block-' + block.id;

    var img = document.createElement('img');
    img.src = window.Api.assetUrl(taskId, block.src);
    img.alt = block.alt_text || '';
    img.onerror = function () {
      handleAssetError(img, block.alt_text || window.I18n.error_asset);
    };
    figure.appendChild(img);

    if (block.caption) {
      var caption = document.createElement('figcaption');
      caption.className = 'block-image__caption';
      caption.textContent = block.caption;
      figure.appendChild(caption);
    }

    return figure;
  }

  function renderAudioBlock(block, taskId) {
    var el = document.createElement('div');
    el.className = 'block block-audio';
    el.id = 'block-' + block.id;

    var audio = document.createElement('audio');
    audio.controls = true;
    audio.preload = 'metadata';
    audio.src = window.Api.assetUrl(taskId, block.src);
    audio.addEventListener('error', function () {
      handleAssetError(audio, window.I18n.error_asset);
    });
    el.appendChild(audio);

    // Transcript is always present on AudioBlock (required field)
    var transcript = document.createElement('div');
    transcript.className = 'block-audio__transcript preserve-whitespace';
    transcript.textContent = block.transcript;
    el.appendChild(transcript);

    return el;
  }

  function renderVideoBlock(block, taskId) {
    var el = document.createElement('div');
    el.className = 'block block-video';
    el.id = 'block-' + block.id;

    var video = document.createElement('video');
    video.controls = true;
    video.preload = 'metadata';
    video.src = window.Api.assetUrl(taskId, block.src);
    if (block.alt_text) {
      video.setAttribute('aria-label', block.alt_text);
    }
    video.addEventListener('error', function () {
      handleAssetError(video, block.alt_text || window.I18n.error_asset);
    });
    el.appendChild(video);

    if (block.transcript) {
      var transcript = document.createElement('div');
      transcript.className = 'block-video__transcript preserve-whitespace';
      transcript.textContent = block.transcript;
      el.appendChild(transcript);
    }

    return el;
  }

  function renderVideoTranscriptBlock(block) {
    var el = document.createElement('div');
    el.className = 'block block-video-transcript';
    el.id = 'block-' + block.id;

    if (block.source_description) {
      var source = document.createElement('div');
      source.className = 'block-video-transcript__source';
      source.textContent = block.source_description;
      el.appendChild(source);
    }

    var transcript = document.createElement('div');
    transcript.className = 'preserve-whitespace';
    transcript.textContent = block.transcript;
    el.appendChild(transcript);

    return el;
  }

  // --------------------------------------------------------------------------
  // Generic Fallback — for unknown block types
  // --------------------------------------------------------------------------

  function renderGenericBlock(block) {
    var el = document.createElement('div');
    el.className = 'block block-generic';
    el.id = 'block-' + block.id;

    var label = document.createElement('div');
    label.className = 'block-generic__label';
    label.textContent = block.type;
    el.appendChild(label);

    // GenericBlock has a `data` dict — display it as formatted JSON
    var dataContent = block.data || {};
    var pre = document.createElement('pre');
    pre.textContent = JSON.stringify(dataContent, null, 2);
    el.appendChild(pre);

    return el;
  }

  // --------------------------------------------------------------------------
  // Renderer Dispatch Map
  // --------------------------------------------------------------------------

  var RENDERERS = {
    text: renderTextBlock,
    image: renderImageBlock,
    audio: renderAudioBlock,
    video: renderVideoBlock,
    video_transcript: renderVideoTranscriptBlock
    // 4b adds: chat_message, social_post, meme
    // 4c adds: search_result (+ replaces generic fallback)
  };

  // --------------------------------------------------------------------------
  // Public API
  // --------------------------------------------------------------------------

  /**
   * Register a new block type renderer.
   * Used by phases 4b and 4c to extend the dispatch map.
   */
  function registerType(type, fn) {
    RENDERERS[type] = fn;
  }

  /**
   * Render a single block. Returns an HTMLElement.
   * Falls through to generic fallback for unknown types.
   */
  function renderBlock(block, taskId) {
    var renderer = RENDERERS[block.type] || renderGenericBlock;
    return renderer(block, taskId);
  }

  /**
   * Render an array of blocks into a DocumentFragment.
   * The caller appends the fragment to the content panel.
   */
  function renderBlocks(blocks, taskId) {
    var fragment = document.createDocumentFragment();
    for (var i = 0; i < blocks.length; i++) {
      fragment.appendChild(renderBlock(blocks[i], taskId));
    }
    return fragment;
  }

  /**
   * Convenience: clear container and render blocks into it.
   * Saves Phase 5a a few lines.
   */
  function renderBlocksInto(container, blocks, taskId) {
    container.innerHTML = '';
    container.appendChild(renderBlocks(blocks, taskId));
  }

  window.Renderer = {
    renderBlocks: renderBlocks,
    renderBlock: renderBlock,
    renderMarkdown: renderMarkdown,
    registerType: registerType,
    renderBlocksInto: renderBlocksInto
  };

})();
