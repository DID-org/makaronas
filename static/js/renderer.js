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
  // Social / Media Block Renderers (Phase 4b)
  // --------------------------------------------------------------------------

  function renderChatMessageBlock(block) {
    var el = document.createElement('div');
    el.className = 'block block-chat-message';
    if (block.is_highlighted) {
      el.className += ' block-chat-message--highlighted';
    }
    el.id = 'block-' + block.id;

    var header = document.createElement('div');
    header.className = 'block-chat-message__header';

    var username = document.createElement('span');
    username.className = 'block-chat-message__username';
    username.textContent = block.username;
    header.appendChild(username);

    if (block.timestamp) {
      var ts = document.createElement('span');
      ts.className = 'block-chat-message__timestamp';
      ts.textContent = block.timestamp;
      header.appendChild(ts);
    }

    el.appendChild(header);

    var body = document.createElement('div');
    body.className = 'block-chat-message__text';
    body.textContent = block.text;
    el.appendChild(body);

    return el;
  }

  function renderSocialPostBlock(block, taskId) {
    var el = document.createElement('div');
    el.className = 'block block-social-post';
    if (block.platform_hint) {
      el.className += ' block-social-post--' + block.platform_hint;
    }
    el.id = 'block-' + block.id;

    var author = document.createElement('div');
    author.className = 'block-social-post__author';
    author.textContent = block.author;
    el.appendChild(author);

    var body = document.createElement('div');
    body.className = 'block-social-post__text';
    body.innerHTML = renderMarkdown(block.text);
    el.appendChild(body);

    if (block.image && taskId) {
      var imgWrap = document.createElement('div');
      imgWrap.className = 'block-social-post__image-wrap';
      var img = document.createElement('img');
      img.className = 'block-social-post__image';
      img.src = window.Api.assetUrl(taskId, block.image);
      img.alt = block.image_alt || '';
      img.onerror = function () {
        handleAssetError(img, block.image_alt || window.I18n.error_asset);
      };
      img.addEventListener('click', function () {
        openLightbox(img.src, img.alt);
      });
      imgWrap.appendChild(img);
      el.appendChild(imgWrap);
    }

    if (block.engagement) {
      var metrics = document.createElement('div');
      metrics.className = 'block-social-post__engagement';
      var entries = Object.entries(block.engagement);
      for (var i = 0; i < entries.length; i++) {
        var metric = document.createElement('span');
        metric.className = 'block-social-post__metric';
        metric.textContent = entries[i][0] + ': ' + entries[i][1];
        metrics.appendChild(metric);
      }
      el.appendChild(metrics);
    }

    if (block.cited_source) {
      var source = document.createElement('div');
      source.className = 'block-social-post__source';
      source.textContent = block.cited_source;
      el.appendChild(source);
    }

    return el;
  }

  function renderMemeBlock(block, taskId) {
    var el = document.createElement('div');
    el.className = 'block block-meme';
    el.id = 'block-' + block.id;

    var img = document.createElement('img');
    img.className = 'block-meme__image';
    img.src = window.Api.assetUrl(taskId, block.image_src);
    img.alt = block.alt_text || '';
    img.onerror = function () {
      handleAssetError(img, block.alt_text || window.I18n.error_asset);
    };
    el.appendChild(img);

    if (block.top_text) {
      var top = document.createElement('div');
      top.className = 'block-meme__text block-meme__text--top';
      top.textContent = block.top_text;
      el.appendChild(top);
    }

    if (block.bottom_text) {
      var bottom = document.createElement('div');
      bottom.className = 'block-meme__text block-meme__text--bottom';
      bottom.textContent = block.bottom_text;
      el.appendChild(bottom);
    }

    if (block.audio_description) {
      var desc = document.createElement('span');
      desc.className = 'sr-only';
      desc.textContent = block.audio_description;
      el.appendChild(desc);
    }

    return el;
  }

  // --------------------------------------------------------------------------
  // Search Result Block Renderer (Phase 4c)
  // --------------------------------------------------------------------------

  function renderSearchResultBlock(block) {
    var el = document.createElement('div');
    el.className = 'block block-search-result';
    if (block.is_key_finding) {
      el.className += ' block-search-result--key-finding';
    }
    if (block.is_dead_end) {
      el.className += ' block-search-result--dead-end';
    }
    el.id = 'block-' + block.id;
    el.setAttribute('role', 'treeitem');

    // Tree nodes with children get aria-expanded; leaf nodes do not
    var hasChildren = block.child_queries && block.child_queries.length > 0;
    if (hasChildren) {
      el.setAttribute('aria-expanded', 'false');
    }

    var query = document.createElement('div');
    query.className = 'block-search-result__query';
    query.textContent = block.query;
    el.appendChild(query);

    var title = document.createElement('div');
    title.className = 'block-search-result__title';
    title.textContent = block.title;
    el.appendChild(title);

    var snippet = document.createElement('div');
    snippet.className = 'block-search-result__snippet';
    snippet.textContent = block.snippet;
    el.appendChild(snippet);

    if (block.url) {
      var link = document.createElement('a');
      link.className = 'block-search-result__url';
      link.href = block.url;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      link.textContent = block.url;
      el.appendChild(link);
    }

    // Screen reader status labels for key finding / dead end
    if (block.is_key_finding) {
      var srKey = document.createElement('span');
      srKey.className = 'sr-only';
      srKey.textContent = window.I18n.search_key_finding;
      el.appendChild(srKey);
    }
    if (block.is_dead_end) {
      var srDead = document.createElement('span');
      srDead.className = 'sr-only';
      srDead.textContent = window.I18n.search_dead_end;
      el.appendChild(srDead);
    }

    // Child group container — empty until Phase 5c populates it
    if (hasChildren) {
      var children = document.createElement('div');
      children.className = 'block-search-result__children';
      children.setAttribute('role', 'group');
      children.setAttribute('data-child-queries', JSON.stringify(block.child_queries));
      children.hidden = true;
      el.appendChild(children);
    }

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

    // Render data as a definition list for readability
    var dataContent = block.data || {};
    var entries = Object.entries(dataContent);

    if (entries.length === 0) {
      var pre = document.createElement('pre');
      pre.textContent = '{}';
      el.appendChild(pre);
    } else {
      var dl = document.createElement('dl');
      dl.className = 'block-generic__data';
      for (var i = 0; i < entries.length; i++) {
        var dt = document.createElement('dt');
        dt.textContent = entries[i][0];
        dl.appendChild(dt);

        var dd = document.createElement('dd');
        var val = entries[i][1];
        if (val === null || val === undefined) {
          dd.textContent = String(val);
        } else if (typeof val === 'object') {
          var valPre = document.createElement('pre');
          valPre.textContent = JSON.stringify(val, null, 2);
          dd.appendChild(valPre);
        } else {
          dd.textContent = String(val);
        }
        dl.appendChild(dd);
      }
      el.appendChild(dl);
    }

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
    video_transcript: renderVideoTranscriptBlock,
    chat_message: renderChatMessageBlock,
    social_post: renderSocialPostBlock,
    meme: renderMemeBlock,
    search_result: renderSearchResultBlock
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

  // --------------------------------------------------------------------------
  // Lightbox — click-to-enlarge for images
  // --------------------------------------------------------------------------

  function openLightbox(src, alt) {
    var overlay = document.getElementById('lightbox-overlay');
    if (!overlay) {
      overlay = document.createElement('div');
      overlay.id = 'lightbox-overlay';
      overlay.className = 'lightbox-overlay';
      overlay.addEventListener('click', function () {
        overlay.classList.remove('lightbox-overlay--visible');
      });
      var img = document.createElement('img');
      img.className = 'lightbox-overlay__image';
      overlay.appendChild(img);
      document.body.appendChild(overlay);
    }
    var img = overlay.querySelector('img');
    img.src = src;
    img.alt = alt || '';
    overlay.classList.add('lightbox-overlay--visible');
  }

  window.Renderer = {
    renderBlocks: renderBlocks,
    renderBlock: renderBlock,
    renderMarkdown: renderMarkdown,
    registerType: registerType,
    renderBlocksInto: renderBlocksInto
  };

})();
