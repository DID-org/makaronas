/* ==========================================================================
   Makaronas — dialogue.js
   Freeform chat UI: message display, streaming, typing indicator, auto-scroll,
   draft persistence, error/redact handling. Reusable streaming display helper.
   Phase 5b foundation + Phase 6a polish.
   ========================================================================== */

(function () {
  'use strict';

  // --------------------------------------------------------------------------
  // Internal State (lives here, not in App state — plan §4.3)
  // --------------------------------------------------------------------------

  var exchanges = [];       // {role, content} pairs for the current phase
  var currentAbort = null;  // Abort handle from streamRespond
  var lastStudentMessage = null; // For retry on error
  var rateLimitRetryCount = 0;   // Rate limit auto-retry tracker (Phase 7b)

  // DOM references (set during render, cleared on cleanup)
  var dialogueArea = null;
  var typingIndicator = null; // Owned by streamDisplay, referenced for bubble insertion
  var inputArea = null;
  var textarea = null;
  var sendBtn = null;
  var streamDisplay = null;   // StreamingDisplay instance for the dialogue area

  // Exchange counter
  var exchangeCounter = null; // DOM element
  var maxExchanges = null;    // From interaction.max_exchanges
  var minExchanges = 3;       // From interaction.min_exchanges (cartridge config)
  var failsafeTarget = null;  // Phase ID from ai_transitions.on_success
  // Emergency-only failsafe: only appears if a student is hopelessly stuck.
  // Flash + the evaluator should drive transitions normally; this is a safety net.
  var FAILSAFE_THRESHOLD = 30;

  // Draft persistence key — same string as STORAGE_KEYS.interactionState in app.js
  // (STORAGE_KEYS is private to app.js IIFE — scout brief §1 confirmed)
  var DRAFT_KEY = 'makaronas_interaction_state';

  // Scroll threshold for manual-scroll-pause detection (px)
  var SCROLL_THRESHOLD = 50;

  // --------------------------------------------------------------------------
  // Shared Helpers (used by both dialogue module and createStreamingDisplay)
  // --------------------------------------------------------------------------

  /**
   * Builds a typing indicator element (animated dots).
   */
  function buildTypingIndicator() {
    var indicator = document.createElement('div');
    indicator.className = 'dialogue-typing';
    indicator.setAttribute('role', 'status');
    var label = document.createElement('span');
    label.className = 'sr-only';
    label.textContent = (window.I18n && window.I18n.dialogue_typing) || 'Typing...';
    var dotsContainer = document.createElement('span');
    dotsContainer.className = 'dialogue-typing__dots';
    for (var i = 0; i < 3; i++) {
      var dot = document.createElement('span');
      dot.className = 'dialogue-typing__dot';
      dotsContainer.appendChild(dot);
    }
    indicator.appendChild(label);
    indicator.appendChild(dotsContainer);
    return indicator;
  }

  /**
   * Replaces streaming text with markdown-rendered HTML using a subtle
   * opacity transition for visual "settling" (Vision §3.7).
   * Double requestAnimationFrame ensures the browser paints the intermediate
   * opacity state before swapping content (Gotcha §6.1).
   */
  function settleMarkdown(bubble, fullText) {
    if (!window.Renderer) {
      bubble.textContent = fullText;
      return;
    }
    bubble.classList.add('dialogue-message--settling');
    bubble.style.opacity = '0.7';
    requestAnimationFrame(function () {
      requestAnimationFrame(function () {
        bubble.innerHTML = window.Renderer.renderMarkdown(fullText);
        bubble.style.opacity = '1';
        function onEnd() {
          bubble.classList.remove('dialogue-message--settling');
          bubble.style.opacity = '';
          bubble.removeEventListener('transitionend', onEnd);
        }
        bubble.addEventListener('transitionend', onEnd);
      });
    });
  }

  /**
   * Appends a redaction notice below a redacted bubble (P12 — honest reveals).
   * Notice auto-fades visually after 5s but remains in the DOM for screen readers.
   */
  function appendRedactionNotice(bubble) {
    var notice = document.createElement('div');
    notice.className = 'dialogue-redaction-notice';
    notice.setAttribute('role', 'alert');
    notice.textContent = (window.I18n && window.I18n.redaction_notice) ||
      'Response was corrected for safety reasons.';
    // Insert notice right after the bubble
    if (bubble.nextSibling) {
      bubble.parentNode.insertBefore(notice, bubble.nextSibling);
    } else {
      bubble.parentNode.appendChild(notice);
    }
  }

  // --------------------------------------------------------------------------
  // Streaming Display Helper (reusable — Phase 6b consumes this)
  // --------------------------------------------------------------------------

  /**
   * Creates a streaming display controller for a container element.
   * Manages typing indicator, token streaming, markdown settling,
   * redaction feedback, and error handling.
   *
   * @param {HTMLElement} container - DOM element to render into
   * @param {Object} options - Configuration
   * @param {Function} options.onComplete - Called after done: (fullText, data)
   * @param {Function} options.onRedact - Called after redaction: (fallbackText)
   * @param {Function} options.onError - Called after error: (code, msg, partial)
   * @param {boolean} options.showTypingIndicator - Show typing dots (default: true)
   * @param {string} options.bubbleClass - CSS class for bubbles
   *   (default: 'dialogue-message dialogue-message--trickster')
   * @returns {Object} Controller: handleToken, handleDone, handleRedact,
   *   handleError, showTyping, getTypingIndicator, destroy
   */
  function createStreamingDisplay(container, options) {
    options = options || {};
    var onComplete = options.onComplete || null;
    var onRedactCb = options.onRedact || null;
    var onErrorCb = options.onError || null;
    var showTypingOpt = options.showTypingIndicator !== false;
    var bubbleClass = options.bubbleClass ||
      'dialogue-message dialogue-message--trickster';

    var sdTypingIndicator = null;
    var sdStreamingBubble = null;
    var sdUserScrolledAway = false;
    var sdStreaming = false;
    var destroyed = false;

    // Create typing indicator
    if (showTypingOpt) {
      sdTypingIndicator = buildTypingIndicator();
      container.appendChild(sdTypingIndicator);
    }

    // Scroll detection for auto-scroll pause
    function onScroll() {
      if (sdStreaming) {
        var atBottom = container.scrollTop + container.clientHeight >=
                       container.scrollHeight - SCROLL_THRESHOLD;
        sdUserScrolledAway = !atBottom;
      }
    }
    container.addEventListener('scroll', onScroll);

    function sdScrollToBottom() {
      container.scrollTop = container.scrollHeight;
      // Mobile: container is in page flow (overflow: visible), scroll page instead
      if (window.innerWidth <= 800) {
        var target = sdStreamingBubble || sdTypingIndicator || container.lastElementChild;
        if (target) target.scrollIntoView({ block: 'end', behavior: 'auto' });
      }
    }

    function sdShowTyping(show) {
      if (!sdTypingIndicator) return;
      if (show) {
        sdTypingIndicator.classList.add('dialogue-typing--active');
      } else {
        sdTypingIndicator.classList.remove('dialogue-typing--active');
      }
    }

    function appendBubble(content, rendered) {
      var bubble = document.createElement('div');
      bubble.className = bubbleClass;
      if (rendered && window.Renderer) {
        bubble.innerHTML = window.Renderer.renderMarkdown(content);
      } else {
        bubble.textContent = content;
      }
      if (sdTypingIndicator && sdTypingIndicator.parentNode === container) {
        container.insertBefore(bubble, sdTypingIndicator);
      } else {
        container.appendChild(bubble);
      }
      return bubble;
    }

    function handleToken(text) {
      if (destroyed) return;
      sdShowTyping(false);
      sdStreaming = true;
      if (!sdStreamingBubble) {
        sdStreamingBubble = appendBubble('', false);
      }
      sdStreamingBubble.textContent += text;
      if (!sdUserScrolledAway) sdScrollToBottom();
    }

    function handleDone(fullText, data) {
      if (destroyed) return;
      sdShowTyping(false);
      sdStreaming = false;
      sdUserScrolledAway = false;
      if (sdStreamingBubble) {
        settleMarkdown(sdStreamingBubble, fullText);
      }
      sdStreamingBubble = null;
      sdScrollToBottom();
      if (onComplete) onComplete(fullText, data);
    }

    function handleRedact(fallbackText) {
      if (destroyed) return;
      sdShowTyping(false);
      sdStreaming = false;
      if (sdStreamingBubble) {
        if (window.Renderer) {
          sdStreamingBubble.innerHTML = window.Renderer.renderMarkdown(fallbackText);
        } else {
          sdStreamingBubble.textContent = fallbackText;
        }
        sdStreamingBubble.classList.add('dialogue-message--redacted');
        appendRedactionNotice(sdStreamingBubble);
      }
      sdStreamingBubble = null;
      sdScrollToBottom();
      if (onRedactCb) onRedactCb(fallbackText);
    }

    function handleError(code, msg, partial) {
      if (destroyed) return;
      sdShowTyping(false);
      sdStreaming = false;
      if (sdStreamingBubble && partial) {
        sdStreamingBubble.textContent = partial;
      } else if (sdStreamingBubble && !sdStreamingBubble.textContent) {
        if (sdStreamingBubble.parentNode) {
          sdStreamingBubble.parentNode.removeChild(sdStreamingBubble);
        }
      }
      sdStreamingBubble = null;
      if (onErrorCb) onErrorCb(code, msg, partial);
    }

    function destroy() {
      destroyed = true;
      sdStreaming = false;
      container.removeEventListener('scroll', onScroll);
      if (sdTypingIndicator && sdTypingIndicator.parentNode) {
        sdTypingIndicator.parentNode.removeChild(sdTypingIndicator);
      }
      sdTypingIndicator = null;
      sdStreamingBubble = null;
    }

    return {
      handleToken: handleToken,
      handleDone: handleDone,
      handleRedact: handleRedact,
      handleError: handleError,
      showTyping: sdShowTyping,
      getTypingIndicator: function () { return sdTypingIndicator; },
      destroy: destroy
    };
  }

  // --------------------------------------------------------------------------
  // Public API
  // --------------------------------------------------------------------------

  /**
   * Renders the complete chat UI into the interaction panel.
   * Called by Interactions.renderInteraction() on type: "freeform".
   * Panel already has heading re-attached by interactions.js.
   */
  function renderFreeformInteraction(panel, interaction, phaseData) {
    // Add flex column class for dialogue layout
    panel.classList.add('interaction-panel--dialogue');

    // Store exchange limits for counter and failsafe
    maxExchanges = (interaction && interaction.max_exchanges) || null;
    minExchanges = (interaction && interaction.min_exchanges) || 3;
    // Store the AI transition target for the failsafe advance button
    var aiTransitions = phaseData.ai_transitions || {};
    failsafeTarget = aiTransitions.on_success || aiTransitions.on_partial || null;

    // Build the chat area
    dialogueArea = document.createElement('div');
    dialogueArea.className = 'dialogue-area';
    dialogueArea.setAttribute('role', 'log');
    dialogueArea.setAttribute('aria-live', 'polite');

    // Build exchange counter (only if max_exchanges is set)
    if (maxExchanges) {
      renderExchangeCounter();
    }

    // Create streaming display (manages typing indicator, streaming, settling)
    streamDisplay = createStreamingDisplay(dialogueArea, {
      onComplete: function (fullText, data) {
        currentAbort = null;
        rateLimitRetryCount = 0;

        // Static fallback detection (Phase 7b — Framework P10)
        if (data && data.fallback === true) {
          // Show notice bubble — honest about AI being unavailable
          showNoticeBubble(
            (window.I18n && window.I18n.ai_fallback_notice) ||
            'AI temporarily unavailable'
          );
          // If backend included next phase content (future-proof), transition
          if (data.next_phase_content) {
            window.App.handlePhaseTransition(data.next_phase_content);
          } else {
            // Re-enable input so student can retry or skip
            setInputDisabled(false);
            if (textarea && window.innerWidth > 800) textarea.focus();
          }
          scrollToBottom();
          return;
        }

        exchanges.push({ role: 'trickster', content: fullText });

        // Update exchange counter from backend authority
        if (data && data.exchanges_count != null) {
          updateExchangeCount(data.exchanges_count);
        }

        // Phase transition or continue
        if (data && data.next_phase_content) {
          var nextPhase = data.next_phase_content;
          if (nextPhase.is_terminal) {
            // Terminal phases transition immediately (reveal flow preserves dialogue)
            window.App.handlePhaseTransition(nextPhase);
          } else {
            // Non-terminal: show continue button so student can read Makaronas's last message
            setInputDisabled(true);
            var continueBtn = document.createElement('button');
            continueBtn.className = 'btn dialogue-continue-btn';
            continueBtn.type = 'button';
            continueBtn.textContent = (window.I18n && window.I18n.btn_continue) || 'T\u0119sti';
            continueBtn.addEventListener('click', function () {
              window.App.handlePhaseTransition(nextPhase);
            });
            // Append button after the last trickster bubble
            var dialogueAreaEl = dialogueArea || document.querySelector('.dialogue-area');
            if (dialogueAreaEl) {
              dialogueAreaEl.appendChild(continueBtn);
            }
          }
        } else {
          setInputDisabled(false);
          if (textarea && window.innerWidth > 800) textarea.focus();

          // Failsafe: only after FAILSAFE_THRESHOLD exchanges (emergency only).
          // Flash + the evaluator handle normal transitions.
          if (data && data.exchanges_count >= FAILSAFE_THRESHOLD) {
            var existing = dialogueArea && dialogueArea.querySelector('.dialogue-failsafe-btn');
            if (!existing && dialogueArea) {
              var failsafeBtn = document.createElement('button');
              failsafeBtn.className = 'btn btn-secondary dialogue-failsafe-btn';
              failsafeBtn.type = 'button';
              failsafeBtn.textContent = (window.I18n && window.I18n.btn_continue) || 'T\u0119sti';
              failsafeBtn.addEventListener('click', function () {
                if (!failsafeTarget) return;
                var sessionId = window.App.getState().session && window.App.getState().session.session_id;
                if (sessionId) {
                  window.Api.submitChoice(sessionId, failsafeTarget, 'Student advanced manually').then(function (phaseData) {
                    window.App.handlePhaseTransition(phaseData);
                  }).catch(function (err) {
                    console.warn('[Makaronas] Failsafe advance failed:', err);
                  });
                }
              });
              dialogueArea.appendChild(failsafeBtn);
            }
          }
        }
        scrollToBottom();
      },
      onRedact: function (fallbackText) {
        currentAbort = null;
        exchanges.push({ role: 'trickster', content: fallbackText });
        setInputDisabled(false);
        if (textarea && window.innerWidth > 800) textarea.focus();
        scrollToBottom();
      },
      onError: function (code, msg, partial) {
        currentAbort = null;

        // Session-fatal errors — route to full error section (Plan §6.2)
        if (code === 'UNAUTHORIZED' || code === 'SESSION_NOT_FOUND') {
          window.App.updateState({ error: { message: msg } });
          return;
        }

        // Rate limit — auto-retry once after cooldown (Plan §6.2)
        if (code === 'RATE_LIMITED' && rateLimitRetryCount < 1) {
          rateLimitRetryCount++;
          showStatusNotice(
            (window.I18n && window.I18n.rate_limit_retry) ||
            'Please wait \u2014 retrying\u2026'
          );
          // Keep input disabled during cooldown
          setTimeout(function () {
            removeStatusNotice();
            retryLastMessage();
          }, 5000);
          scrollToBottom();
          return;
        }

        // Rate limit exhausted — show as normal error, re-enable input
        if (code === 'RATE_LIMITED') {
          rateLimitRetryCount = 0;
          showErrorDisplay(msg ||
            ((window.I18n && window.I18n.error_rate_limit) || 'Please wait.'));
          setInputDisabled(false);
          scrollToBottom();
          return;
        }

        // All other errors — show with retry + skip actions (Plan §6.3)
        showErrorDisplay(msg ||
          ((window.I18n && window.I18n.dialogue_error_partial) ||
           'Response interrupted.'), {
          actions: [
            {
              label: (window.I18n && window.I18n.dialogue_error_retry) || 'Retry',
              handler: function () { retryLastMessage(); }
            },
            {
              label: (window.I18n && window.I18n.error_skip_task) || 'Skip task',
              handler: function () { window.App.skipCurrentTask(); }
            }
          ]
        });
        setInputDisabled(false);
        scrollToBottom();
      }
    });

    // Get typing indicator reference for bubble insertion
    typingIndicator = streamDisplay.getTypingIndicator();

    // Build the input area
    inputArea = document.createElement('div');
    inputArea.className = 'dialogue-input-area';

    textarea = document.createElement('textarea');
    textarea.className = 'dialogue-input';
    textarea.setAttribute('aria-label', (window.I18n && window.I18n.placeholder_message) || 'Type your response...');
    textarea.setAttribute('placeholder', (window.I18n && window.I18n.placeholder_message) || 'Type your response...');
    textarea.rows = 2;

    sendBtn = document.createElement('button');
    sendBtn.className = 'btn dialogue-send-btn';
    sendBtn.type = 'button';
    sendBtn.textContent = (window.I18n && window.I18n.btn_send) || 'Send';

    inputArea.appendChild(textarea);
    inputArea.appendChild(sendBtn);

    // Append to panel (below heading)
    panel.appendChild(dialogueArea);
    panel.appendChild(inputArea);

    // Render trickster opening message (§6.5: use phaseData.trickster_intro)
    var tricksterIntro = phaseData.trickster_intro;
    if (tricksterIntro) {
      appendTricksterBubble(tricksterIntro, true);
      exchanges.push({ role: 'trickster', content: tricksterIntro });
    }

    // Restore draft text if phase matches
    restoreDraft(phaseData.current_phase);

    // Draft persistence on input
    textarea.addEventListener('input', function () {
      saveDraft(phaseData.current_phase);
    });

    // Send handlers
    sendBtn.addEventListener('click', function () {
      submitMessage(phaseData);
    });

    textarea.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        submitMessage(phaseData);
      }
    });

    // If app is already locked (e.g., recovery mid-stream), disable input
    var appState = window.App.getState();
    if (appState.locked) {
      setInputDisabled(true);
    }
  }

  /**
   * Re-renders an array of {role, content} exchange objects as chat bubbles.
   * Used by session recovery in app.js.
   */
  function renderDialogueHistory(panel, history) {
    if (!history || !history.length) return;
    if (!dialogueArea) return;

    // Clear existing messages (keep typing indicator and exchange counter)
    var children = dialogueArea.childNodes;
    for (var i = children.length - 1; i >= 0; i--) {
      if (children[i] !== typingIndicator && children[i] !== exchangeCounter) {
        dialogueArea.removeChild(children[i]);
      }
    }

    // Re-render exchanges
    exchanges = [];
    for (var j = 0; j < history.length; j++) {
      var exchange = history[j];
      if (exchange.role === 'trickster') {
        appendTricksterBubble(exchange.content, true);
      } else if (exchange.role === 'student') {
        appendStudentBubble(exchange.content);
      }
      exchanges.push({ role: exchange.role, content: exchange.content });
    }

    // Ensure typing indicator is at the end
    dialogueArea.appendChild(typingIndicator);

    // Update exchange counter from recovered history (Gotcha §6.2)
    if (exchangeCounter && maxExchanges) {
      var studentCount = 0;
      for (var k = 0; k < exchanges.length; k++) {
        if (exchanges[k].role === 'student') studentCount++;
      }
      updateExchangeCount(studentCount);
    }

    scrollToBottom();
  }

  /**
   * Resets internal dialogue state. Called on phase transitions
   * to prevent stale state from leaking across phases.
   */
  function clearDialogue() {
    exchanges = [];
    lastStudentMessage = null;
    rateLimitRetryCount = 0;

    if (currentAbort) {
      currentAbort.abort();
      currentAbort = null;
    }

    // Destroy streaming display (cleans up typing indicator, scroll listener)
    if (streamDisplay) {
      streamDisplay.destroy();
      streamDisplay = null;
    }

    // Remove dialogue layout class if present
    var panel = document.querySelector('.interaction-panel');
    if (panel) {
      panel.classList.remove('interaction-panel--dialogue');
    }

    // Clear DOM references
    dialogueArea = null;
    typingIndicator = null;
    inputArea = null;
    textarea = null;
    sendBtn = null;
    exchangeCounter = null;
    maxExchanges = null;
  }

  // --------------------------------------------------------------------------
  // Message Rendering
  // --------------------------------------------------------------------------

  /**
   * Creates and appends a trickster message bubble.
   * If rendered=true, content is markdown-rendered via innerHTML.
   * Otherwise content is plain text via textContent (used during streaming).
   */
  function appendTricksterBubble(content, rendered) {
    var bubble = document.createElement('div');
    bubble.className = 'dialogue-message dialogue-message--trickster';

    if (rendered && window.Renderer) {
      bubble.innerHTML = window.Renderer.renderMarkdown(content);
    } else {
      bubble.textContent = content;
    }

    // Insert before typing indicator
    if (typingIndicator && typingIndicator.parentNode === dialogueArea) {
      dialogueArea.insertBefore(bubble, typingIndicator);
    } else {
      dialogueArea.appendChild(bubble);
    }

    return bubble;
  }

  /**
   * Creates and appends a student message bubble.
   * Always textContent — student input is untrusted (P13).
   */
  function appendStudentBubble(content) {
    var bubble = document.createElement('div');
    bubble.className = 'dialogue-message dialogue-message--student';
    bubble.textContent = content;

    // Insert before typing indicator
    if (typingIndicator && typingIndicator.parentNode === dialogueArea) {
      dialogueArea.insertBefore(bubble, typingIndicator);
    } else {
      dialogueArea.appendChild(bubble);
    }

    scrollToBottom();
    return bubble;
  }

  // --------------------------------------------------------------------------
  // Exchange Counter
  // --------------------------------------------------------------------------

  /**
   * Creates the exchange counter element at the top of the dialogue area.
   * Sticky positioned so it remains visible while scrolling.
   */
  function renderExchangeCounter() {
    exchangeCounter = document.createElement('div');
    exchangeCounter.className = 'dialogue-exchange-counter';
    exchangeCounter.setAttribute('aria-label',
      (window.I18n && window.I18n.exchange_counter_label) || 'Conversation progress');
    var text = document.createElement('span');
    text.className = 'dialogue-exchange-counter__text';
    text.textContent = formatExchangeCount(0, maxExchanges);
    exchangeCounter.appendChild(text);
    // Insert as first child of dialogueArea
    if (dialogueArea.firstChild) {
      dialogueArea.insertBefore(exchangeCounter, dialogueArea.firstChild);
    } else {
      dialogueArea.appendChild(exchangeCounter);
    }
  }

  /**
   * Updates the exchange counter text with the current count.
   */
  function updateExchangeCount(count) {
    if (!exchangeCounter || !maxExchanges) return;
    var text = exchangeCounter.querySelector('.dialogue-exchange-counter__text');
    if (text) {
      text.textContent = formatExchangeCount(count, maxExchanges);
    }
  }

  function formatExchangeCount(current, max) {
    var template = (window.I18n && window.I18n.exchange_counter) || '{current}/{max}';
    return template.replace('{current}', current).replace('{max}', max);
  }

  // --------------------------------------------------------------------------
  // Streaming
  // --------------------------------------------------------------------------

  /**
   * Submits the student's message to the Trickster via SSE streaming.
   */
  function submitMessage(phaseData) {
    if (!textarea || !dialogueArea) return;

    var text = textarea.value.trim();
    if (!text) return;

    var appState = window.App.getState();
    if (appState.locked) return;
    if (!appState.session) return;

    var sessionId = appState.session.session_id;
    lastStudentMessage = text;

    // Remove any previous error display
    removeErrorDisplay();

    // Add student bubble
    appendStudentBubble(text);
    exchanges.push({ role: 'student', content: text });

    // Clear textarea and draft
    textarea.value = '';
    clearDraft();

    // Mobile scroll-to-top guard:
    // On iOS Safari, disabling a focused textarea dismisses the keyboard and
    // can yank window.scrollY to 0 (body becomes the active element). Blur
    // explicitly first so keyboard dismissal is deterministic, then restore
    // scroll position after one frame if iOS moved us away from where we were.
    var isMobile = window.innerWidth <= 800;
    var savedScrollY = isMobile ? window.scrollY : null;
    if (isMobile && textarea) {
      textarea.blur();
    }

    // Disable input during streaming
    setInputDisabled(true);

    if (isMobile && savedScrollY !== null) {
      requestAnimationFrame(function () {
        if (Math.abs(window.scrollY - savedScrollY) > 100) {
          window.scrollTo(0, savedScrollY);
        }
      });
    }

    // Show typing indicator via streaming display
    if (streamDisplay) {
      streamDisplay.showTyping(true);
    }

    // Start streaming — wire directly to streaming display callbacks
    var result = window.Api.streamRespond(sessionId, 'respond', text, {
      onToken: streamDisplay.handleToken,
      onDone: streamDisplay.handleDone,
      onRedact: streamDisplay.handleRedact,
      onError: streamDisplay.handleError
    });

    currentAbort = result;
  }

  // --------------------------------------------------------------------------
  // UI Helpers
  // --------------------------------------------------------------------------

  function setInputDisabled(disabled) {
    if (textarea) {
      textarea.disabled = disabled;
    }
    if (sendBtn) {
      sendBtn.disabled = disabled;
      if (disabled) {
        sendBtn.title = (window.I18n && window.I18n.dialogue_send_disabled) || 'Wait for response';
      } else {
        sendBtn.title = '';
      }
    }
  }

  function scrollToBottom() {
    if (!dialogueArea) return;
    dialogueArea.scrollTop = dialogueArea.scrollHeight;
    // Mobile: dialogueArea is in page flow, scroll page to show input area.
    // Use 'auto' (instant) — 'smooth' races with iOS keyboard-dismiss reflow
    // when textarea.disabled fires immediately after, and the browser can
    // cancel the animation mid-flight and resolve to the page top.
    if (window.innerWidth <= 800 && inputArea) {
      inputArea.scrollIntoView({ block: 'end', behavior: 'auto' });
    }
  }

  /**
   * Shows an inline error/status message in the dialogue area.
   *
   * @param {string} message - Display text
   * @param {Object} [options] - Configuration
   * @param {Array} [options.actions] - [{label: string, handler: function}]
   * @param {boolean} [options.isStatus] - Use status styling instead of error
   */
  function showErrorDisplay(message, options) {
    removeErrorDisplay();
    if (!dialogueArea) return;

    options = options || {};
    var cssClass = options.isStatus ? 'dialogue-status' : 'dialogue-error';

    var errorEl = document.createElement('div');
    errorEl.className = cssClass;

    var msgSpan = document.createElement('span');
    msgSpan.textContent = message;
    errorEl.appendChild(msgSpan);

    // Action buttons — custom or default retry
    var actions = options.actions;
    if (!actions && !options.isStatus) {
      actions = [{
        label: (window.I18n && window.I18n.dialogue_error_retry) || 'Retry',
        handler: function () { retryLastMessage(); }
      }];
    }

    if (actions && actions.length > 0) {
      var actionsDiv = document.createElement('div');
      actionsDiv.className = 'dialogue-error-actions';
      for (var i = 0; i < actions.length; i++) {
        var btn = document.createElement('button');
        btn.className = 'dialogue-error-actions__btn';
        btn.type = 'button';
        btn.textContent = actions[i].label;
        btn.addEventListener('click', actions[i].handler);
        actionsDiv.appendChild(btn);
      }
      errorEl.appendChild(actionsDiv);
    }

    // Insert before typing indicator
    if (typingIndicator && typingIndicator.parentNode === dialogueArea) {
      dialogueArea.insertBefore(errorEl, typingIndicator);
    } else {
      dialogueArea.appendChild(errorEl);
    }
  }

  /**
   * Removes error and status displays to prevent stacking (Plan §8.6).
   */
  function removeErrorDisplay() {
    if (!dialogueArea) return;
    var existing = dialogueArea.querySelectorAll('.dialogue-error, .dialogue-status');
    for (var i = 0; i < existing.length; i++) {
      existing[i].parentNode.removeChild(existing[i]);
    }
  }

  /**
   * Retries the last student message — removes the student's exchange
   * from the array (the backend already has it, but we re-send so the
   * student doesn't need to re-type).
   */
  function retryLastMessage() {
    if (!lastStudentMessage) return;

    var appState = window.App.getState();
    if (appState.locked) return;
    if (!appState.session) return;

    removeErrorDisplay();
    setInputDisabled(true);

    if (streamDisplay) {
      streamDisplay.showTyping(true);
    }

    var sessionId = appState.session.session_id;

    var result = window.Api.streamRespond(sessionId, 'respond', lastStudentMessage, {
      onToken: streamDisplay.handleToken,
      onDone: streamDisplay.handleDone,
      onRedact: streamDisplay.handleRedact,
      onError: streamDisplay.handleError
    });

    currentAbort = result;
  }

  // --------------------------------------------------------------------------
  // Notice / Status Helpers (Phase 7b)
  // --------------------------------------------------------------------------

  /**
   * Shows a system notice bubble in the dialogue area — not from trickster
   * or student. Used for fallback notices (Framework P10, P2).
   */
  function showNoticeBubble(text) {
    if (!dialogueArea) return;
    var notice = document.createElement('div');
    notice.className = 'dialogue-message dialogue-message--notice';
    notice.textContent = text;
    if (typingIndicator && typingIndicator.parentNode === dialogueArea) {
      dialogueArea.insertBefore(notice, typingIndicator);
    } else {
      dialogueArea.appendChild(notice);
    }
  }

  /**
   * Shows a transient status notice in the dialogue area (rate limit cooldown).
   */
  function showStatusNotice(text) {
    removeStatusNotice();
    if (!dialogueArea) return;
    var notice = document.createElement('div');
    notice.className = 'dialogue-status';
    var span = document.createElement('span');
    span.textContent = text;
    notice.appendChild(span);
    if (typingIndicator && typingIndicator.parentNode === dialogueArea) {
      dialogueArea.insertBefore(notice, typingIndicator);
    } else {
      dialogueArea.appendChild(notice);
    }
  }

  /**
   * Removes status notices from the dialogue area.
   */
  function removeStatusNotice() {
    if (!dialogueArea) return;
    var existing = dialogueArea.querySelectorAll('.dialogue-status');
    for (var i = 0; i < existing.length; i++) {
      existing[i].parentNode.removeChild(existing[i]);
    }
  }

  // --------------------------------------------------------------------------
  // Draft Persistence (§4.4)
  // --------------------------------------------------------------------------

  function saveDraft(phaseId) {
    if (!textarea) return;
    try {
      sessionStorage.setItem(DRAFT_KEY, JSON.stringify({
        phase_id: phaseId,
        draft: textarea.value
      }));
    } catch (e) {
      // sessionStorage unavailable or full — silently fail
    }
  }

  function restoreDraft(phaseId) {
    if (!textarea) return;
    try {
      var stored = sessionStorage.getItem(DRAFT_KEY);
      if (!stored) return;
      var parsed = JSON.parse(stored);
      if (parsed && parsed.phase_id === phaseId && parsed.draft) {
        textarea.value = parsed.draft;
      }
    } catch (e) {
      // Parse error or sessionStorage unavailable — ignore
    }
  }

  function clearDraft() {
    try {
      sessionStorage.removeItem(DRAFT_KEY);
    } catch (e) {
      // Ignore
    }
  }

  // --------------------------------------------------------------------------
  // Public API
  // --------------------------------------------------------------------------

  window.Dialogue = {
    renderFreeformInteraction: renderFreeformInteraction,
    renderDialogueHistory: renderDialogueHistory,
    clearDialogue: clearDialogue,
    createStreamingDisplay: createStreamingDisplay,
    appendStudentBubble: appendStudentBubble
  };

})();
