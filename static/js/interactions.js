/* ==========================================================================
   Makaronas — interactions.js
   Interaction dispatcher: renders controls in the interaction panel based on
   the phase's interaction config. Phase 5a handles "button" type.
   Phase 5b adds "freeform", Phase 5c adds "investigation".
   Phase 7c adds "generation" (empathy flip) detection before freeform.
   ========================================================================== */

(function () {
  'use strict';

  var GENERATION_TIMEOUT_MS = 15000;
  var DRAFT_KEY = 'makaronas_interaction_state';

  /**
   * Renders the appropriate interaction controls into the interaction panel.
   * Dispatches on phaseData.interaction.type. Clears the panel first.
   *
   * Generation phases are detected via available_actions (not interaction.type).
   * A generation phase has type "freeform" but available_actions includes "generate".
   * If dialogue history exists, the student already passed through generation —
   * fall through to the normal freeform case.
   */
  function renderInteraction(phaseData) {
    var panel = document.querySelector('.interaction-panel');
    if (!panel) return;

    // Preserve the heading, clear everything else
    var heading = panel.querySelector('h2');
    panel.innerHTML = '';
    if (heading) {
      panel.appendChild(heading);
    }

    var interaction = phaseData.interaction;
    if (!interaction) {
      // No interaction — panel stays empty (content-only phase)
      return;
    }

    // Generation detection: available_actions includes "generate" AND no
    // dialogue history (meaning the student hasn't sent to Trickster yet).
    // Scout brief §4 confirmed: check App.getState().dialogueHistory for recovery.
    if (phaseData.available_actions &&
        phaseData.available_actions.indexOf('generate') !== -1) {
      var appDialogueHistory = window.App.getState().dialogueHistory;
      if (!appDialogueHistory || appDialogueHistory.length === 0) {
        renderGenerationInteraction(panel, interaction, phaseData);
        return;
      }
      // Dialogue history exists — student already passed through generation.
      // Fall through to freeform case below.
    }

    switch (interaction.type) {
      case 'button':
        renderButtonInteraction(panel, interaction, phaseData);
        break;
      case 'freeform':
        if (window.Dialogue) {
          window.Dialogue.renderFreeformInteraction(panel, interaction, phaseData);
        }
        break;
      case 'investigation':
        if (window.Investigation) {
          var contentPanel = document.querySelector('.content-panel');
          window.Investigation.renderInvestigation(contentPanel, panel, interaction, phaseData);
        }
        break;
      default:
        renderUnsupportedInteraction(panel, interaction.type);
        break;
    }
  }

  // --------------------------------------------------------------------------
  // Generation Interaction (Phase 7c — Empathy Flip)
  // --------------------------------------------------------------------------

  /**
   * Races a promise against a timeout. Rejects with {isTimeout: true} on expiry.
   */
  function withTimeout(promise, ms) {
    return Promise.race([
      promise,
      new Promise(function (_, reject) {
        setTimeout(function () { reject({ isTimeout: true }); }, ms);
      })
    ]);
  }

  /**
   * Extracts text content from phase presentation blocks for the /generate
   * source_content parameter.
   */
  function extractSourceContent(phaseData) {
    var parts = [];
    var blocks = phaseData.content || [];
    for (var i = 0; i < blocks.length; i++) {
      var block = blocks[i];
      if (block.type === 'text' && block.data && block.data.text) {
        parts.push(block.data.text);
      }
    }
    return parts.join('\n\n');
  }

  /**
   * Saves the generation prompt draft to sessionStorage.
   */
  function saveGenerationDraft(phaseId, text) {
    try {
      sessionStorage.setItem(DRAFT_KEY, JSON.stringify({
        type: 'generation',
        phase_id: phaseId,
        draft: text
      }));
    } catch (e) {
      // sessionStorage unavailable — ignore
    }
  }

  /**
   * Restores generation prompt draft from sessionStorage if phase matches.
   */
  function restoreGenerationDraft(phaseId) {
    try {
      var raw = sessionStorage.getItem(DRAFT_KEY);
      if (!raw) return '';
      var parsed = JSON.parse(raw);
      if (parsed.type === 'generation' && parsed.phase_id === phaseId) {
        return parsed.draft || '';
      }
    } catch (e) {
      // Parse error or unavailable — ignore
    }
    return '';
  }

  /**
   * Clears the generation draft from sessionStorage.
   */
  function clearGenerationDraft() {
    try {
      sessionStorage.removeItem(DRAFT_KEY);
    } catch (e) {
      // Ignore
    }
  }

  /**
   * Renders the empathy flip generation UI (Stage 1).
   * Student writes a prompt → POST /generate → view result → send to Trickster.
   */
  function renderGenerationInteraction(panel, interaction, phaseData) {
    var container = document.createElement('div');
    container.className = 'generation-container';

    // --- Prompt area ---
    var promptArea = document.createElement('div');
    promptArea.className = 'generation-prompt-area';

    var label = document.createElement('label');
    label.className = 'generation-label';
    label.textContent = (window.I18n && window.I18n.generation_prompt_label) ||
      'Write instructions for the AI:';
    label.setAttribute('for', 'generation-textarea');
    promptArea.appendChild(label);

    var textarea = document.createElement('textarea');
    textarea.className = 'generation-input';
    textarea.id = 'generation-textarea';
    textarea.placeholder = (window.I18n && window.I18n.placeholder_generation) ||
      'Write your manipulation attempt\u2026';
    textarea.rows = 5;

    // Restore draft
    var restoredDraft = restoreGenerationDraft(phaseData.current_phase);
    if (restoredDraft) {
      textarea.value = restoredDraft;
    }

    // Draft persistence on input
    textarea.addEventListener('input', function () {
      saveGenerationDraft(phaseData.current_phase, textarea.value);
    });

    promptArea.appendChild(textarea);
    container.appendChild(promptArea);

    // --- Actions ---
    var actionsArea = document.createElement('div');
    actionsArea.className = 'generation-actions';

    var generateBtn = document.createElement('button');
    generateBtn.className = 'btn generation-submit-btn';
    generateBtn.type = 'button';
    generateBtn.textContent = (window.I18n && window.I18n.btn_generate) || 'Generate';
    actionsArea.appendChild(generateBtn);

    container.appendChild(actionsArea);

    // --- Result area (hidden initially) ---
    var resultArea = document.createElement('div');
    resultArea.className = 'generation-result';
    resultArea.setAttribute('aria-live', 'polite');
    resultArea.style.display = 'none';
    container.appendChild(resultArea);

    // --- Error area ---
    var errorArea = document.createElement('div');
    errorArea.className = 'generation-error';
    errorArea.style.display = 'none';
    container.appendChild(errorArea);

    panel.appendChild(container);

    // Focus textarea
    textarea.focus();

    // Track rate limit retry state
    var rateLimitRetried = false;

    // --- Generate button handler ---
    generateBtn.addEventListener('click', function () {
      var prompt = textarea.value.trim();
      if (!prompt) {
        textarea.focus();
        return;
      }

      var appState = window.App.getState();
      if (!appState.session) return;
      var sessionId = appState.session.session_id;

      // Loading state
      generateBtn.disabled = true;
      generateBtn.setAttribute('aria-busy', 'true');
      generateBtn.textContent = (window.I18n && window.I18n.generation_loading) || 'Generating\u2026';
      textarea.disabled = true;
      errorArea.style.display = 'none';
      resultArea.style.display = 'none';

      var sourceContent = extractSourceContent(phaseData);

      withTimeout(
        window.Api.generate(sessionId, sourceContent, prompt),
        GENERATION_TIMEOUT_MS
      ).then(function (result) {
        // Clear draft — generation succeeded
        clearGenerationDraft();

        // Show result
        showGenerationResult(resultArea, prompt, result, panel, interaction, phaseData);
        resultArea.style.display = '';

        // Hide prompt area and actions, keep result visible
        promptArea.style.display = 'none';
        actionsArea.style.display = 'none';

      }).catch(function (err) {
        // Re-enable input
        generateBtn.disabled = false;
        generateBtn.removeAttribute('aria-busy');
        generateBtn.textContent = (window.I18n && window.I18n.btn_generate) || 'Generate';
        textarea.disabled = false;

        if (err.isTimeout) {
          showGenerationError(errorArea,
            (window.I18n && window.I18n.generation_timeout) ||
            'Generation took too long. Try again.');
        } else if (err.statusCode === 429 && !rateLimitRetried) {
          // Rate limit — auto-retry once after 5s
          rateLimitRetried = true;
          showGenerationError(errorArea,
            (window.I18n && window.I18n.rate_limit_retry) ||
            'Please wait \u2014 retrying\u2026');
          generateBtn.disabled = true;
          textarea.disabled = true;

          setTimeout(function () {
            generateBtn.disabled = false;
            textarea.disabled = false;
            errorArea.style.display = 'none';
            // Programmatic retry
            generateBtn.click();
          }, 5000);
        } else if (err.code === 'UNAUTHORIZED' || err.code === 'SESSION_NOT_FOUND') {
          // Session-fatal
          window.App.updateState({ error: { message: err.message } });
        } else if (err.code === 'AI_UNAVAILABLE') {
          showGenerationError(errorArea,
            (window.I18n && window.I18n.generation_error) ||
            'Generation failed. Try again.',
            [
              {
                label: (window.I18n && window.I18n.btn_retry) || 'Retry',
                handler: function () {
                  errorArea.style.display = 'none';
                  generateBtn.click();
                }
              },
              {
                label: (window.I18n && window.I18n.error_skip_task) || 'Skip task',
                handler: function () {
                  window.App.skipCurrentTask();
                }
              }
            ]);
        } else {
          // Network or unexpected error
          showGenerationError(errorArea,
            (err.message) ||
            (window.I18n && window.I18n.generation_error) ||
            'Generation failed. Try again.');
        }
      });
    });
  }

  /**
   * Shows an error message within the generation container.
   */
  function showGenerationError(errorArea, message, actions) {
    errorArea.innerHTML = '';
    errorArea.style.display = '';

    var msgSpan = document.createElement('span');
    msgSpan.textContent = message;
    errorArea.appendChild(msgSpan);

    if (actions && actions.length > 0) {
      var actionsDiv = document.createElement('div');
      actionsDiv.className = 'generation-error-actions';
      for (var i = 0; i < actions.length; i++) {
        var btn = document.createElement('button');
        btn.className = 'btn generation-error-actions__btn';
        btn.type = 'button';
        btn.textContent = actions[i].label;
        btn.addEventListener('click', actions[i].handler);
        actionsDiv.appendChild(btn);
      }
      errorArea.appendChild(actionsDiv);
    }
  }

  /**
   * Renders the generation result: student's prompt, AI output, safety notice,
   * and "Siųsti Tricksteriui" button. The send button triggers Stage 2 transition.
   */
  function showGenerationResult(resultArea, studentPrompt, result, panel, interaction, phaseData) {
    resultArea.innerHTML = '';

    // Student's prompt (reference)
    var promptSection = document.createElement('div');
    promptSection.className = 'generation-result__prompt';
    var promptLabel = document.createElement('div');
    promptLabel.className = 'generation-result__label';
    promptLabel.textContent = (window.I18n && window.I18n.generation_your_prompt) ||
      'Your instruction';
    promptSection.appendChild(promptLabel);
    var promptText = document.createElement('div');
    promptText.className = 'generation-result__prompt-text';
    promptText.textContent = studentPrompt;
    promptSection.appendChild(promptText);
    resultArea.appendChild(promptSection);

    // AI-generated output
    var outputSection = document.createElement('div');
    outputSection.className = 'generation-result__output';
    var outputHeading = document.createElement('h3');
    outputHeading.className = 'generation-result__heading';
    outputHeading.textContent = (window.I18n && window.I18n.generation_result_heading) ||
      'Generated result';
    outputSection.appendChild(outputHeading);

    var outputContent = document.createElement('div');
    outputContent.className = 'generation-result__content';
    // Render through markdown + DOMPurify pipeline (P12, P13)
    if (window.Renderer) {
      outputContent.innerHTML = window.Renderer.renderMarkdown(result.generated_text);
    } else {
      outputContent.textContent = result.generated_text;
    }
    outputSection.appendChild(outputContent);
    resultArea.appendChild(outputSection);

    // Safety notice (if content was redacted)
    if (result.safety_redacted === true) {
      var safetyNotice = document.createElement('p');
      safetyNotice.className = 'generation-safety-notice';
      safetyNotice.textContent = (window.I18n && window.I18n.generation_safety_notice) ||
        'Content was adjusted for safety reasons.';
      resultArea.appendChild(safetyNotice);
    }

    // "Siųsti Tricksteriui" button — THE action (P1: Train the Pause)
    var sendActionsArea = document.createElement('div');
    sendActionsArea.className = 'generation-result__actions';
    var sendBtn = document.createElement('button');
    sendBtn.className = 'btn generation-send-btn';
    sendBtn.type = 'button';
    sendBtn.textContent = (window.I18n && window.I18n.btn_submit_generation) ||
      'Send to Trickster';
    sendActionsArea.appendChild(sendBtn);
    resultArea.appendChild(sendActionsArea);

    // Focus the send button for keyboard users
    sendBtn.focus();

    // Stage 2 transition: send to Trickster
    sendBtn.addEventListener('click', function () {
      transitionToDialogue(panel, interaction, phaseData, result.generated_text);
    });
  }

  /**
   * Stage 2 transition: replaces generation UI with freeform dialogue,
   * then programmatically submits the generated text as the first message.
   *
   * Approach: set up the dialogue UI via renderFreeformInteraction, then set
   * the textarea value and trigger the send button click. This reuses the
   * dialogue module's own submit flow (bubble creation, exchange tracking,
   * streaming, phase transitions) without duplicating any logic.
   */
  function transitionToDialogue(panel, interaction, phaseData, generatedText) {
    if (!window.Dialogue) return;

    var appState = window.App.getState();
    if (!appState.session) return;

    // Clear the generation container from the panel (preserve heading)
    var heading = panel.querySelector('h2');
    panel.innerHTML = '';
    if (heading) {
      panel.appendChild(heading);
    }

    // Set up the standard freeform dialogue UI
    window.Dialogue.renderFreeformInteraction(panel, interaction, phaseData);

    // Set the textarea value and programmatically trigger submit.
    // submitMessage() reads textarea.value, creates the student bubble,
    // pushes to exchanges, and starts streaming — all handled internally.
    var dialogueTextarea = panel.querySelector('.dialogue-input');
    var dialogueSendBtn = panel.querySelector('.dialogue-send-btn');

    if (dialogueTextarea) {
      dialogueTextarea.value = generatedText;
    }

    if (dialogueSendBtn) {
      dialogueSendBtn.click();
    }
  }

  // --------------------------------------------------------------------------
  // Button Interaction (Phase 5a)
  // --------------------------------------------------------------------------

  /**
   * Renders button choices as styled, accessible buttons.
   * Uses a single delegated click handler on the container.
   */
  function renderButtonInteraction(panel, interaction, phaseData) {
    var container = document.createElement('div');
    container.className = 'choice-container';
    container.setAttribute('role', 'group');
    container.setAttribute('aria-label', (window.I18n && window.I18n.interaction_heading) || 'Choices');

    var choices = interaction.choices || [];
    var appState = window.App.getState();

    for (var i = 0; i < choices.length; i++) {
      var choice = choices[i];
      var btn = document.createElement('button');
      btn.className = 'btn choice-btn';
      btn.type = 'button';
      btn.textContent = choice.label;
      btn.dataset.targetPhase = choice.target_phase;
      if (choice.context_label) {
        btn.dataset.contextLabel = choice.context_label;
      }
      if (appState.locked) {
        btn.disabled = true;
      }
      container.appendChild(btn);
    }

    // Delegated click handler on the container
    container.addEventListener('click', function (e) {
      var btn = e.target.closest('.choice-btn');
      if (!btn || btn.disabled) return;

      var targetPhase = btn.dataset.targetPhase;
      var contextLabel = btn.dataset.contextLabel || null;
      var sessionId = window.App.getState().session && window.App.getState().session.session_id;

      if (!sessionId) return;

      // Disable all buttons during submission
      var allBtns = container.querySelectorAll('.choice-btn');
      for (var j = 0; j < allBtns.length; j++) {
        allBtns[j].disabled = true;
      }

      Api.submitChoice(sessionId, targetPhase, contextLabel).then(function (data) {
        window.App.handlePhaseTransition(data);
      }).catch(function (err) {
        // Re-enable buttons on error
        for (var j = 0; j < allBtns.length; j++) {
          allBtns[j].disabled = false;
        }
        window.App.updateState({ error: { message: err.message } });
      });
    });

    panel.appendChild(container);
  }

  /**
   * Renders a fallback message for unsupported interaction types.
   * Graceful degradation — shows type name and Lithuanian message.
   */
  function renderUnsupportedInteraction(panel, typeName) {
    var notice = document.createElement('p');
    notice.className = 'interaction-unsupported';
    notice.textContent = (window.I18n && window.I18n.interaction_unsupported) || 'Unsupported interaction type';
    panel.appendChild(notice);
  }

  // --------------------------------------------------------------------------
  // Public API
  // --------------------------------------------------------------------------

  window.Interactions = {
    renderInteraction: renderInteraction
  };

})();
