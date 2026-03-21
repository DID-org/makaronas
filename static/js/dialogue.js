/* ==========================================================================
   Makaronas — dialogue.js
   Freeform chat UI: message display, streaming, typing indicator, auto-scroll,
   draft persistence, error/redact handling.
   Phase 5b — where the Trickster comes alive.
   ========================================================================== */

(function () {
  'use strict';

  // --------------------------------------------------------------------------
  // Internal State (lives here, not in App state — plan §4.3)
  // --------------------------------------------------------------------------

  var exchanges = [];       // {role, content} pairs for the current phase
  var streaming = false;    // True while SSE stream is active
  var currentAbort = null;  // Abort handle from streamRespond
  var userScrolledAway = false;
  var lastStudentMessage = null; // For retry on error

  // DOM references (set during render, cleared on cleanup)
  var dialogueArea = null;
  var typingIndicator = null;
  var inputArea = null;
  var textarea = null;
  var sendBtn = null;
  var streamingBubble = null;

  // Draft persistence key — same string as STORAGE_KEYS.interactionState in app.js
  // (STORAGE_KEYS is private to app.js IIFE — scout brief §1 confirmed)
  var DRAFT_KEY = 'makaronas_interaction_state';

  // Scroll threshold for manual-scroll-pause detection (px)
  var SCROLL_THRESHOLD = 50;

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

    // Build the chat area
    dialogueArea = document.createElement('div');
    dialogueArea.className = 'dialogue-area';
    dialogueArea.setAttribute('role', 'log');
    dialogueArea.setAttribute('aria-live', 'polite');

    // Typing indicator (hidden by default)
    typingIndicator = document.createElement('div');
    typingIndicator.className = 'dialogue-typing';
    typingIndicator.setAttribute('role', 'status');
    var typingLabel = document.createElement('span');
    typingLabel.className = 'sr-only';
    typingLabel.textContent = (window.I18n && window.I18n.dialogue_typing) || 'Typing...';
    var dotsContainer = document.createElement('span');
    dotsContainer.className = 'dialogue-typing__dots';
    for (var i = 0; i < 3; i++) {
      var dot = document.createElement('span');
      dot.className = 'dialogue-typing__dot';
      dotsContainer.appendChild(dot);
    }
    typingIndicator.appendChild(typingLabel);
    typingIndicator.appendChild(dotsContainer);
    dialogueArea.appendChild(typingIndicator);

    // Auto-scroll detection
    dialogueArea.addEventListener('scroll', function () {
      if (streaming) {
        var atBottom = dialogueArea.scrollTop + dialogueArea.clientHeight >=
                       dialogueArea.scrollHeight - SCROLL_THRESHOLD;
        userScrolledAway = !atBottom;
      }
    });

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

    // Clear existing messages (but keep typing indicator)
    var children = dialogueArea.childNodes;
    for (var i = children.length - 1; i >= 0; i--) {
      if (children[i] !== typingIndicator) {
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
    scrollToBottom();
  }

  /**
   * Resets internal dialogue state. Called on phase transitions
   * to prevent stale state from leaking across phases.
   */
  function clearDialogue() {
    exchanges = [];
    streaming = false;
    userScrolledAway = false;
    lastStudentMessage = null;
    streamingBubble = null;

    if (currentAbort) {
      currentAbort.abort();
      currentAbort = null;
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

    // Disable input during streaming
    setInputDisabled(true);

    // Show typing indicator
    showTyping(true);

    // Start streaming
    streaming = true;
    userScrolledAway = false;
    streamingBubble = null;

    var result = window.Api.streamRespond(sessionId, 'respond', text, {
      onToken: handleToken,
      onDone: function (fullText, data) {
        handleDone(fullText, data, phaseData);
      },
      onRedact: handleRedact,
      onError: handleError
    });

    currentAbort = result;
  }

  /**
   * Handles a streaming token — appends text to the current bubble.
   */
  function handleToken(text) {
    // Hide typing indicator on first token
    showTyping(false);

    // Create streaming bubble if needed
    if (!streamingBubble) {
      streamingBubble = appendTricksterBubble('', false);
    }

    // Append via textContent (safe, fast — §4.1)
    streamingBubble.textContent += text;

    // Auto-scroll unless student scrolled away
    if (!userScrolledAway) {
      scrollToBottom();
    }
  }

  /**
   * Handles the done event — finalizes the trickster response.
   */
  function handleDone(fullText, data) {
    showTyping(false);
    streaming = false;
    currentAbort = null;

    // Replace streaming bubble content with markdown-rendered version (§4.1)
    if (streamingBubble && window.Renderer) {
      streamingBubble.innerHTML = window.Renderer.renderMarkdown(fullText);
    }

    exchanges.push({ role: 'trickster', content: fullText });
    streamingBubble = null;

    // Check for phase transition (§6.4)
    if (data && data.next_phase_content) {
      // Phase transition — handlePhaseTransition will call renderPhase
      // which calls clearDialogue via the wiring in app.js
      window.App.handlePhaseTransition(data.next_phase_content);
    } else {
      // Conversation continues — re-enable input
      setInputDisabled(false);
      if (textarea) {
        textarea.focus();
      }
    }

    scrollToBottom();
  }

  /**
   * Handles redact event — replaces bubble content with fallback (P12).
   */
  function handleRedact(fallbackText) {
    showTyping(false);
    streaming = false;
    currentAbort = null;

    if (streamingBubble) {
      // Replace with sanitised fallback via markdown renderer
      if (window.Renderer) {
        streamingBubble.innerHTML = window.Renderer.renderMarkdown(fallbackText);
      } else {
        streamingBubble.textContent = fallbackText;
      }
      // Visual flash to signal safety correction
      streamingBubble.classList.add('dialogue-message--redacted');
    }

    exchanges.push({ role: 'trickster', content: fallbackText });
    streamingBubble = null;

    // Re-enable input for next turn
    setInputDisabled(false);
    if (textarea) {
      textarea.focus();
    }
    scrollToBottom();
  }

  /**
   * Handles stream error — preserves partial text, shows retry option.
   */
  function handleError(code, message, partialText) {
    showTyping(false);
    streaming = false;
    currentAbort = null;

    // If there's partial text in the streaming bubble, keep it
    if (streamingBubble && partialText) {
      streamingBubble.textContent = partialText;
    } else if (streamingBubble && !streamingBubble.textContent) {
      // Empty streaming bubble — remove it
      if (streamingBubble.parentNode) {
        streamingBubble.parentNode.removeChild(streamingBubble);
      }
    }
    streamingBubble = null;

    // Show inline error with retry
    showErrorDisplay(message || ((window.I18n && window.I18n.dialogue_error_partial) || 'Response interrupted.'));

    // Re-enable input
    setInputDisabled(false);
    scrollToBottom();
  }

  // --------------------------------------------------------------------------
  // UI Helpers
  // --------------------------------------------------------------------------

  function showTyping(show) {
    if (typingIndicator) {
      if (show) {
        typingIndicator.classList.add('dialogue-typing--active');
      } else {
        typingIndicator.classList.remove('dialogue-typing--active');
      }
    }
  }

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
    if (dialogueArea) {
      dialogueArea.scrollTop = dialogueArea.scrollHeight;
    }
  }

  /**
   * Shows an inline error message with a retry button below the last message.
   */
  function showErrorDisplay(message) {
    removeErrorDisplay();
    if (!dialogueArea) return;

    var errorEl = document.createElement('div');
    errorEl.className = 'dialogue-error';

    var msgSpan = document.createElement('span');
    msgSpan.textContent = message;
    errorEl.appendChild(msgSpan);

    var retryBtn = document.createElement('button');
    retryBtn.className = 'dialogue-error__retry';
    retryBtn.type = 'button';
    retryBtn.textContent = (window.I18n && window.I18n.dialogue_error_retry) || 'Retry';
    retryBtn.addEventListener('click', function () {
      retryLastMessage();
    });
    errorEl.appendChild(retryBtn);

    // Insert before typing indicator
    if (typingIndicator && typingIndicator.parentNode === dialogueArea) {
      dialogueArea.insertBefore(errorEl, typingIndicator);
    } else {
      dialogueArea.appendChild(errorEl);
    }
  }

  function removeErrorDisplay() {
    if (!dialogueArea) return;
    var existing = dialogueArea.querySelector('.dialogue-error');
    if (existing) {
      existing.parentNode.removeChild(existing);
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

    // Remove the last student exchange from local array since we're re-sending
    // Actually, the backend processes each /respond independently, so we
    // just re-send. Keep the existing student bubble visible.

    setInputDisabled(true);
    showTyping(true);
    streaming = true;
    userScrolledAway = false;
    streamingBubble = null;

    var sessionId = appState.session.session_id;

    var result = window.Api.streamRespond(sessionId, 'respond', lastStudentMessage, {
      onToken: handleToken,
      onDone: function (fullText, data) {
        handleDone(fullText, data);
      },
      onRedact: handleRedact,
      onError: handleError
    });

    currentAbort = result;
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
    clearDialogue: clearDialogue
  };

})();
