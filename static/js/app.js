/* ==========================================================================
   Makaronas — app.js
   Entry point: central state, updateState/render cycle, section switching.
   ========================================================================== */

(function () {
  'use strict';

  // --------------------------------------------------------------------------
  // State
  // --------------------------------------------------------------------------

  var state = {
    section: 'welcome',    // which section is visible
    error: null,           // { message: string } | null
    locked: false,         // UI lockout during API calls (Phase 3a)
    session: null,         // { session_id: string, auth_token: string } | null
    task: null,            // Full phase data dict from API (_derive_phase_response)
    phase: null,           // Current phase ID string (shorthand for task.current_phase)
    taskSequenceIndex: 0,  // Position in TASK_SEQUENCE array
    terminal: null,        // { evaluation_outcome, reveal } when is_terminal, null otherwise
    dialogueHistory: []    // Exchange history for dialogue recovery
  };

  // --------------------------------------------------------------------------
  // Task Sequence (honest stub — replaced by V9 Roadmap Engine)
  // --------------------------------------------------------------------------

  // Only active cartridges — draft skeletons (cherry-pick, phantom-quote,
  // wedge, misleading-frame) are excluded until archetype visions complete them.
  var TASK_SEQUENCE = [
    'task-vaitkus-001',           // hybrid, investigation — MVP Story Task 1: two articles
    'task-vaitkus-comments-001',  // ai_driven, social — MVP Story Task 2: comment section
    'task-tunguska-001',          // ai_driven, article — rhetoric analysis
    'task-follow-money-001',      // hybrid, investigation — deeper, guided discovery
    'task-clickbait-trap-001'     // hybrid, article — clickbait patterns
  ];

  // --------------------------------------------------------------------------
  // Session Storage Keys (namespaced to avoid collisions)
  // --------------------------------------------------------------------------

  var STORAGE_KEYS = {
    sessionId: 'makaronas_session_id',
    authToken: 'makaronas_auth_token',
    interactionState: 'makaronas_interaction_state',
    taskSequenceIndex: 'makaronas_task_sequence_index'
  };

  // --------------------------------------------------------------------------
  // State Management
  // --------------------------------------------------------------------------

  /**
   * Merges changes into state and triggers a render.
   * All state mutations go through this function.
   */
  function updateState(changes) {
    Object.assign(state, changes);
    // Derived state: error implies error section
    if (state.error && state.section !== 'error') {
      state.section = 'error';
    }
    render();
  }

  /**
   * Returns a shallow copy of state — prevents external mutation.
   */
  function getState() {
    return Object.assign({}, state);
  }

  // --------------------------------------------------------------------------
  // Render
  // --------------------------------------------------------------------------

  /**
   * Master render — reads state and updates DOM.
   * Future phases add conditional render calls here.
   */
  function render() {
    renderSection(state.section);
    renderError(state.error);
  }

  /**
   * Sets the active section via data attribute.
   * CSS handles show/hide — JS only sets the attribute.
   */
  function renderSection(sectionId) {
    var app = document.getElementById('app');
    if (app) {
      app.dataset.activeSection = sectionId;
    }
    // Scroll to top when switching sections
    window.scrollTo(0, 0);

    // Focus management: move focus to the new section's heading.
    // setTimeout(0) ensures the display change has painted before focusing —
    // without this, .focus() silently fails on elements transitioning from
    // display: none in some browsers (especially Safari).
    setTimeout(function () {
      var section = document.getElementById('section-' + sectionId);
      if (section) {
        var heading = section.querySelector('h2[tabindex="-1"]');
        if (heading) {
          heading.focus();
        }
      }
    }, 0);
  }

  /**
   * Updates the error section content.
   * Pure render — reads error state, updates DOM text only.
   */
  function renderError(error) {
    var messageEl = document.getElementById('error-message');
    if (!messageEl) return;

    if (error && error.message) {
      messageEl.textContent = error.message;
    } else if (error) {
      // Error object without message — use Lithuanian generic fallback
      messageEl.textContent = (window.I18n && window.I18n.error_generic) || '';
    } else {
      messageEl.textContent = '';
    }
  }

  // --------------------------------------------------------------------------
  // Phase Rendering + Task Loading
  // --------------------------------------------------------------------------

  /**
   * Returns a Lithuanian context hint for the task medium.
   * Helps the student understand what they're looking at.
   */
  function _getMediumHint(medium) {
    var hints = {
      'article':          'Ka\u017ekas pasidalino \u0161iuo straipsniu socialiniuose tinkluose.',
      'social_post':      '\u0160is \u012fra\u0161as plinta socialiniuose tinkluose.',
      'chat':             'Pokalbis, kur\u012f ka\u017ekas tau persiunt\u0117.',
      'investigation':    'I\u0161tirk ir surask ties\u0105.',
      'image':            'Ka\u017ekas pasidalino \u0161ia nuotrauka.'
    };
    return hints[medium] || null;
  }

  /**
   * Renders a phase into the content and interaction panels.
   * Called on initial task load, phase transitions, and session recovery.
   * This is imperative — not called from the render() loop.
   */
  function renderPhase(phaseData) {
    // Update state with new phase data — clear dialogue history on phase transitions
    // so the debrief only reflects the final AI dialogue, not accumulated history
    updateState({
      task: phaseData,
      phase: phaseData.current_phase,
      section: 'task',
      dialogueHistory: []
    });

    // Clear interaction state from previous phase
    if (window.Dialogue) {
      window.Dialogue.clearDialogue();
    }
    if (window.Investigation) {
      window.Investigation.clearInvestigation();
    }

    // Content panel: task header + presentation blocks
    var contentPanel = document.querySelector('.content-panel');
    if (contentPanel && window.Renderer) {
      console.log('[Makaronas] renderPhase:', phaseData.current_phase, 'blocks:', (phaseData.content || []).map(function(b) { return b.id; }));
      window.Renderer.renderBlocksInto(contentPanel, phaseData.content || [], phaseData.task_id);

      // Build task header: number + title + medium context
      var taskHeader = document.createElement('div');
      taskHeader.className = 'task-header';

      var taskNumber = state.taskSequenceIndex + 1;
      var taskLabel = document.createElement('span');
      taskLabel.className = 'task-header__label';
      taskLabel.textContent = 'U\u017eduotis ' + taskNumber;

      var taskTitle = document.createElement('h2');
      taskTitle.className = 'task-header__title';
      taskTitle.tabIndex = -1;
      taskTitle.textContent = phaseData.title || '';

      taskHeader.appendChild(taskLabel);
      taskHeader.appendChild(taskTitle);

      // Medium context — tell the student what they're looking at
      var mediumHint = _getMediumHint(phaseData.medium);
      if (mediumHint) {
        var contextEl = document.createElement('p');
        contextEl.className = 'task-header__context';
        contextEl.textContent = mediumHint;
        taskHeader.appendChild(contextEl);
      }

      contentPanel.insertBefore(taskHeader, contentPanel.firstChild);
    }

    // Layout class: reset and apply variant based on interaction type
    var taskLayout = document.querySelector('.task-layout');
    if (taskLayout) {
      taskLayout.className = 'task-layout';
      if (phaseData.interaction && phaseData.interaction.type === 'investigation') {
        taskLayout.classList.add('layout-investigation');
      } else if (phaseData.interaction && phaseData.interaction.type === 'button') {
        taskLayout.classList.add('layout-fullpage');
      }
    }

    // Interaction panel: render controls via interactions.js
    if (window.Interactions) {
      window.Interactions.renderInteraction(phaseData);
    }

    // Scroll to top on phase transition and focus management
    setTimeout(function () {
      window.scrollTo(0, 0);
      if (contentPanel) contentPanel.scrollTop = 0;
      var heading = contentPanel && contentPanel.querySelector('h2[tabindex="-1"]');
      if (heading) {
        heading.focus();
      }
    }, 0);
  }

  /**
   * Loads the first task from TASK_SEQUENCE after session creation.
   * Chains from createSessionFlow().
   */
  function loadFirstTask() {
    if (!state.session) return;

    var taskId = TASK_SEQUENCE[state.taskSequenceIndex];
    if (!taskId) {
      updateState({ error: { message: (window.I18n && window.I18n.task_load_error) || 'Task load error' } });
      return;
    }

    Api.loadTask(state.session.session_id, taskId).then(function (data) {
      renderPhase(data);
      // Persist index on successful load
      sessionStorage.setItem(STORAGE_KEYS.taskSequenceIndex, String(state.taskSequenceIndex));
      console.log('[Makaronas] Task loaded:', taskId);
    }).catch(function (err) {
      updateState({ error: { message: err.message || (window.I18n && window.I18n.task_load_error) || 'Task load error' } });
    });
  }

  /**
   * Loads the next task from TASK_SEQUENCE after post-task flow completes.
   * Called by the "Kitas uždavinys" button click handler.
   */
  function loadNextTask() {
    if (state.locked) return;

    // Abort any in-flight debrief stream
    if (debriefAbort) {
      debriefAbort.abort();
      debriefAbort = null;
    }

    // Increment sequence index
    var nextIndex = state.taskSequenceIndex + 1;

    // End of sequence — show session end screen
    if (nextIndex >= TASK_SEQUENCE.length) {
      updateState({ taskSequenceIndex: nextIndex, terminal: null });
      // Persist exhausted index so refresh shows end screen, not last task
      sessionStorage.setItem(STORAGE_KEYS.taskSequenceIndex, String(nextIndex));
      showSessionEnd();
      return;
    }

    // Clean up post-task DOM elements from the interaction panel
    var interactionPanel = document.querySelector('.interaction-panel');
    if (interactionPanel) {
      var postTaskEls = interactionPanel.querySelectorAll(
        '.debrief-section, .reveal-section, .post-task-separator, .post-task-next-btn'
      );
      for (var i = 0; i < postTaskEls.length; i++) {
        postTaskEls[i].parentNode.removeChild(postTaskEls[i]);
      }
    }

    // Reset terminal state before loading new task
    updateState({ taskSequenceIndex: nextIndex, terminal: null });

    var taskId = TASK_SEQUENCE[nextIndex];
    Api.loadTask(state.session.session_id, taskId).then(function (data) {
      renderPhase(data);
      // Persist index on successful load
      sessionStorage.setItem(STORAGE_KEYS.taskSequenceIndex, String(nextIndex));
      console.log('[Makaronas] Next task loaded:', taskId);
    }).catch(function (err) {
      updateState({ error: { message: err.message || (window.I18n && window.I18n.task_load_error) || 'Task load error' } });
    });
  }

  /**
   * Skips the current task and loads the next one in TASK_SEQUENCE.
   * Called by dialogue error handlers when the student chooses to skip (Phase 7b).
   * Delegates to loadNextTask() for full cleanup and next-task logic.
   */
  function skipCurrentTask() {
    loadNextTask();
  }

  /**
   * Shows the session end screen with reflective prompt and start-new option.
   * Called when all tasks in TASK_SEQUENCE have been completed.
   */
  function showSessionEnd() {
    updateState({ section: 'end' });
    console.log('[Makaronas] Session complete — all tasks finished');
  }

  // --------------------------------------------------------------------------
  // Post-Task Flow (Phase 6b — Debrief + Reveal)
  // --------------------------------------------------------------------------

  // Track debrief abort handle so it can be cleaned up on navigation (Phase 7a)
  var debriefAbort = null;

  /**
   * Orchestrates the post-task experience when a terminal phase is reached.
   * Preserves dialogue history, appends debrief (streamed) + reveal (static) +
   * "Kitas uždavinys" button — all as siblings inside the interaction panel.
   */
  function startPostTaskFlow(phaseData) {
    // Update state to stay on 'task' section with the terminal phase data
    updateState({
      task: phaseData,
      phase: phaseData.current_phase,
      section: 'task'
    });

    var interactionPanel = document.querySelector('.interaction-panel');
    if (!interactionPanel) return;

    // 1. Remove the chat input area (keep dialogue history)
    var inputArea = interactionPanel.querySelector('.dialogue-input-area');
    if (inputArea) {
      inputArea.parentNode.removeChild(inputArea);
    }

    // 2. Switch overflow model — remove flex column layout so the panel
    //    scrolls all layers (dialogue + debrief + reveal) together.
    interactionPanel.classList.remove('interaction-panel--dialogue');

    // 3. Handle case where there's no dialogue area (terminal reached via buttons).
    //    Clear the panel of button choices, but keep the heading.
    var dialogueArea = interactionPanel.querySelector('.dialogue-area');
    if (!dialogueArea) {
      var heading = interactionPanel.querySelector('h2');
      interactionPanel.innerHTML = '';
      if (heading) {
        interactionPanel.appendChild(heading);
      }
    }

    // 4. Update content panel if terminal phase has content blocks
    if (phaseData.content && phaseData.content.length > 0) {
      var contentPanel = document.querySelector('.content-panel');
      if (contentPanel && window.Renderer) {
        var contentHeading = contentPanel.querySelector('h2');
        window.Renderer.renderBlocksInto(contentPanel, phaseData.content, phaseData.task_id);
        if (contentHeading) {
          contentHeading.textContent = phaseData.title || ((window.I18n && window.I18n.content_heading) || 'Turinys');
          contentPanel.insertBefore(contentHeading, contentPanel.firstChild);
        }
      }
    }

    // 5. Scroll dialogue to end (if present) so student sees last exchange
    if (dialogueArea) {
      dialogueArea.scrollTop = dialogueArea.scrollHeight;
    }

    // 6. Add visual separator
    var separator = document.createElement('hr');
    separator.className = 'post-task-separator';
    separator.setAttribute('aria-hidden', 'true');
    interactionPanel.appendChild(separator);

    // 7. Start debrief (if there was AI dialogue), then reveal + button
    //    Static terminal phases (reached via buttons, no AI exchanges) skip
    //    the debrief — the trickster_content already contains the reveal.
    var hadDialogue = state.dialogueHistory && state.dialogueHistory.length > 0;
    var sessionId = state.session && state.session.session_id;

    if (!hadDialogue) {
      // Skip debrief — show reveal + next button directly
      _renderPostTaskEnding(interactionPanel);
      return;
    }

    renderDebrief(interactionPanel, sessionId, function () {
      _renderPostTaskEnding(interactionPanel);
    });
  }

  /**
   * Renders reveal section + "Kitas uždavinys" button.
   * Called after debrief completes, or directly for static terminal phases.
   */
  function _renderPostTaskEnding(container) {
    if (state.terminal && state.terminal.reveal) {
      renderReveal(container, state.terminal.reveal);
    }

    var nextBtn = document.createElement('button');
    nextBtn.className = 'btn btn-primary post-task-next-btn';
    nextBtn.textContent = (window.I18n && window.I18n.btn_next_task) || 'Kitas u\u017Edavinys';
    nextBtn.id = 'btn-next-task';
    nextBtn.addEventListener('click', loadNextTask);
    container.appendChild(nextBtn);

    setTimeout(function () { nextBtn.focus(); }, 0);
  }

  /**
   * Creates a debrief container and streams the debrief into it.
   * On completion or error, calls onComplete so the caller can render reveal.
   */
  function renderDebrief(container, sessionId, onComplete) {
    // Create debrief wrapper
    var debriefSection = document.createElement('div');
    debriefSection.className = 'debrief-section';
    debriefSection.setAttribute('aria-label', (window.I18n && window.I18n.debrief_heading) || 'Aptarimas');

    var heading = document.createElement('h3');
    heading.textContent = (window.I18n && window.I18n.debrief_heading) || 'Aptarimas';
    heading.setAttribute('tabindex', '-1');
    debriefSection.appendChild(heading);

    container.appendChild(debriefSection);

    // Smooth-scroll so the debrief heading is visible
    debriefSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Guard: if no session or no streaming infrastructure, skip to reveal
    if (!sessionId || !window.Dialogue || !window.Api) {
      showDebriefSkipNotice(debriefSection);
      if (onComplete) onComplete();
      return;
    }

    var debriefRetryCount = 0;
    var debriefSkipNotice = (window.I18n && window.I18n.debrief_skip_notice) || 'AI nepasiekiamas';

    /**
     * Starts (or restarts) the debrief stream.
     * On retry: clears section content (except heading), creates fresh display.
     */
    function startDebriefStream() {
      // Clear section content except heading (for retry)
      var children = debriefSection.childNodes;
      for (var i = children.length - 1; i >= 0; i--) {
        if (children[i] !== heading) {
          debriefSection.removeChild(children[i]);
        }
      }

      var display = window.Dialogue.createStreamingDisplay(debriefSection, {
        bubbleClass: 'debrief-content',
        showTypingIndicator: true,
        onComplete: function () {
          setTimeout(function () { heading.focus(); }, 0);
          if (onComplete) onComplete();
        },
        onError: function (code, msg, partial) {
          // Rate limit — retry once after cooldown (Phase 7b)
          if (code === 'RATE_LIMITED' && debriefRetryCount < 1) {
            debriefRetryCount++;
            var retryNotice = document.createElement('p');
            retryNotice.className = 'debrief-skip-notice';
            retryNotice.textContent = (window.I18n && window.I18n.rate_limit_retry) ||
              'Please wait \u2014 retrying\u2026';
            debriefSection.appendChild(retryNotice);
            setTimeout(function () {
              startDebriefStream();
            }, 5000);
            return;
          }
          // All other errors or retry exhausted — skip to reveal
          showDebriefSkipNotice(debriefSection);
          if (onComplete) onComplete();
        }
      });

      display.showTyping(true);

      var result = window.Api.streamDebrief(sessionId, {
        onToken: display.handleToken,
        onDone: display.handleDone,
        onRedact: function (fallbackText) {
          display.handleError('REDACTED', debriefSkipNotice, '');
        },
        onError: display.handleError
      });

      debriefAbort = result || null;
    }

    startDebriefStream();
  }

  /**
   * Shows a brief notice when the debrief stream fails or is unavailable.
   */
  function showDebriefSkipNotice(debriefSection) {
    var notice = document.createElement('p');
    notice.className = 'debrief-skip-notice';
    notice.textContent = (window.I18n && window.I18n.debrief_skip_notice) ||
      'AI nepasiekiamas \u2014 \u0161tai svarbiausia pamoka';
    debriefSection.appendChild(notice);
  }

  /**
   * Renders reveal content (key lesson + additional resources) as static HTML.
   * revealData shape: { key_lesson: string, additional_resources: [string, ...] }
   */
  function renderReveal(container, revealData) {
    if (!revealData) return;

    var revealSection = document.createElement('div');
    revealSection.className = 'reveal-section';
    revealSection.setAttribute('aria-label', (window.I18n && window.I18n.reveal_heading) || 'I\u0161vados');

    var heading = document.createElement('h3');
    heading.textContent = (window.I18n && window.I18n.reveal_heading) || 'I\u0161vados';
    heading.setAttribute('tabindex', '-1');
    revealSection.appendChild(heading);

    // Key lesson — rendered through markdown pipeline (trusted cartridge content)
    if (revealData.key_lesson) {
      var lessonDiv = document.createElement('div');
      lessonDiv.className = 'reveal-lesson';
      if (window.Renderer) {
        lessonDiv.innerHTML = window.Renderer.renderMarkdown(revealData.key_lesson);
      } else {
        lessonDiv.textContent = revealData.key_lesson;
      }
      revealSection.appendChild(lessonDiv);
    }

    // Additional resources
    if (revealData.additional_resources && revealData.additional_resources.length > 0) {
      var resourcesDiv = document.createElement('div');
      resourcesDiv.className = 'reveal-resources';

      var resourcesHeading = document.createElement('h4');
      resourcesHeading.textContent = (window.I18n && window.I18n.reveal_resources_heading) || 'Papildomi \u0161altiniai';
      resourcesDiv.appendChild(resourcesHeading);

      for (var i = 0; i < revealData.additional_resources.length; i++) {
        var item = document.createElement('div');
        item.className = 'reveal-resource-item';
        if (window.Renderer) {
          item.innerHTML = window.Renderer.renderMarkdown(revealData.additional_resources[i]);
        } else {
          item.textContent = revealData.additional_resources[i];
        }
        resourcesDiv.appendChild(item);
      }

      revealSection.appendChild(resourcesDiv);
    }

    container.appendChild(revealSection);

    // Smooth-scroll and focus management
    revealSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    setTimeout(function () { heading.focus(); }, 0);
  }

  /**
   * Unified entry point for all phase transitions.
   * Both button clicks (5a) and SSE done events (5b) route here.
   */
  function handlePhaseTransition(phaseData) {
    if (phaseData.is_terminal) {
      // Store terminal data for debrief/reveal
      // Clear dialogue history — the reveal conclusions replace the debrief
      updateState({
        terminal: {
          evaluation_outcome: phaseData.evaluation_outcome,
          reveal: phaseData.reveal
        },
        dialogueHistory: []
      });
      // Terminal fork: preserve dialogue, start post-task flow
      startPostTaskFlow(phaseData);
    } else {
      // Clear stale terminal data from previous phases
      if (state.terminal !== null) {
        updateState({ terminal: null });
      }
      renderPhase(phaseData);
    }
  }

  // --------------------------------------------------------------------------
  // Session Helpers
  // --------------------------------------------------------------------------

  /**
   * Clears all session data from sessionStorage and state.
   * Safe to call even when no session exists (no-op for missing keys).
   */
  function clearSession() {
    sessionStorage.removeItem(STORAGE_KEYS.sessionId);
    sessionStorage.removeItem(STORAGE_KEYS.authToken);
    sessionStorage.removeItem(STORAGE_KEYS.interactionState);
    sessionStorage.removeItem(STORAGE_KEYS.taskSequenceIndex);
    updateState({ session: null, terminal: null, taskSequenceIndex: 0 });
  }

  /**
   * Creates a new backend session and persists credentials.
   * Called by the start button click handler.
   */
  function createSessionFlow() {
    if (state.locked) return;

    var authToken = crypto.randomUUID();

    Api.createSession(authToken).then(function (data) {
      sessionStorage.setItem(STORAGE_KEYS.sessionId, data.session_id);
      sessionStorage.setItem(STORAGE_KEYS.authToken, authToken);
      updateState({ session: { session_id: data.session_id, auth_token: authToken } });
      console.log('[Makaronas] Session created:', data.session_id);
      loadFirstTask();
    }).catch(function (err) {
      updateState({ error: { message: err.message } });
    });
  }

  /**
   * Attempts to recover an existing session from sessionStorage.
   * Called once during init() — if credentials exist, queries the backend
   * for current session state and restores it.
   */
  function recoverSession() {
    var sessionId;
    var authToken;

    try {
      sessionId = sessionStorage.getItem(STORAGE_KEYS.sessionId);
      authToken = sessionStorage.getItem(STORAGE_KEYS.authToken);
    } catch (e) {
      // sessionStorage unavailable (rare: old Safari private browsing, quota)
      // Fall through to welcome screen
      return;
    }

    if (!sessionId || !authToken) return;

    // Recover taskSequenceIndex from sessionStorage
    var storedIndex = sessionStorage.getItem(STORAGE_KEYS.taskSequenceIndex);
    var recoveredIndex = storedIndex !== null ? parseInt(storedIndex, 10) : 0;
    if (isNaN(recoveredIndex)) recoveredIndex = 0;

    // Populate state BEFORE API call — getCurrentSession reads
    // state.session.auth_token for the Bearer header (see IMPL_NOTES 3a)
    updateState({ session: { session_id: sessionId, auth_token: authToken }, taskSequenceIndex: recoveredIndex });

    // If stored index indicates all tasks exhausted, go straight to end screen
    if (recoveredIndex >= TASK_SEQUENCE.length) {
      showSessionEnd();
      console.log('[Makaronas] Recovery: task sequence exhausted, showing end screen');
      return;
    }

    Api.getCurrentSession(sessionId).then(function (data) {
      if (data.current_task === null) {
        // Session alive but no active task — show welcome
        console.log('[Makaronas] Recovery: session alive, no active task');
        return;
      }

      // Reconcile taskSequenceIndex with actual current task from backend
      var backendIndex = TASK_SEQUENCE.indexOf(data.task_id);
      if (backendIndex >= 0 && backendIndex !== state.taskSequenceIndex) {
        updateState({ taskSequenceIndex: backendIndex });
      }

      // Store dialogue history for potential recovery
      updateState({ dialogueHistory: data.dialogue_history || [] });

      // Terminal phase recovery: route through handlePhaseTransition
      // which calls startPostTaskFlow (debrief restarts from scratch)
      if (data.is_terminal) {
        // Render the phase first to set up content + interaction panels
        renderPhase(data);

        // Restore dialogue history if available (freeform terminal)
        if (state.dialogueHistory.length > 0 && window.Dialogue) {
          var recoveryPanel = document.querySelector('.interaction-panel');
          if (recoveryPanel) {
            window.Dialogue.renderDialogueHistory(recoveryPanel, state.dialogueHistory);
          }
        }

        // Now trigger terminal flow (stores terminal data, starts post-task)
        handlePhaseTransition(data);
        console.log('[Makaronas] Recovery: restored terminal phase, restarting post-task flow');
        return;
      }

      // Non-terminal: render the current phase
      renderPhase(data);

      // Restore dialogue history if the current phase is a freeform interaction (Phase 5b)
      if (state.dialogueHistory.length > 0 && window.Dialogue) {
        var recoveryPanel = document.querySelector('.interaction-panel');
        if (recoveryPanel) {
          window.Dialogue.renderDialogueHistory(recoveryPanel, state.dialogueHistory);
        }
      }

      // Restore investigation state if the current phase is an investigation interaction (Phase 5c)
      if (data.interaction && data.interaction.type === 'investigation' && window.Investigation) {
        var recoveryContent = document.querySelector('.content-panel');
        var recoveryInteraction = document.querySelector('.interaction-panel');
        if (recoveryContent && recoveryInteraction) {
          window.Investigation.recoverInvestigation(recoveryContent, recoveryInteraction, data);
        }
      }

      console.log('[Makaronas] Recovery: restored active task');
    }).catch(function (err) {
      if (err.code === 'SESSION_NOT_FOUND' || err.code === 'TASK_CONTENT_UPDATED') {
        // Session expired or task changed — clean start
        clearSession();
        console.log('[Makaronas] Recovery:', err.code, '— cleared session');
        return;
      }
      // Network or unexpected error — clear session, show error
      clearSession();
      updateState({ error: { message: err.message } });
    });
  }

  // --------------------------------------------------------------------------
  // Event Binding
  // --------------------------------------------------------------------------

  function init() {
    // Welcome: start button — creates backend session
    var startBtn = document.getElementById('btn-start');
    if (startBtn) {
      startBtn.addEventListener('click', createSessionFlow);
    }

    // Error: retry button — return to welcome, clear session + error
    var retryBtn = document.getElementById('btn-retry');
    if (retryBtn) {
      retryBtn.addEventListener('click', function () {
        clearSession();
        updateState({ section: 'welcome', error: null });
      });
    }

    // Session End: "Pradėti iš naujo" button — clears session, returns to welcome
    var startNewBtn = document.getElementById('btn-start-new');
    if (startNewBtn) {
      startNewBtn.addEventListener('click', function () {
        clearSession();
        updateState({ section: 'welcome', error: null });
      });
    }

    // DEV: Skip button — jumps to next task in sequence
    var devSkipBtn = document.getElementById('btn-dev-skip');
    if (devSkipBtn) {
      devSkipBtn.addEventListener('click', function () {
        if (!state.session) return;
        console.log('[DEV] Skipping task at index', state.taskSequenceIndex);
        loadNextTask();
      });
    }

    // DEV: Write button — skips to write_article phase via /choice API
    var devWriteBtn = document.getElementById('btn-dev-write');
    if (devWriteBtn) {
      devWriteBtn.addEventListener('click', function () {
        if (!state.session) return;
        var sessionId = state.session.session_id;
        console.log('[DEV] Skipping to write_article phase');
        Api.submitChoice(sessionId, 'write_article', 'DEV: skipped to write_article').then(function (data) {
          handlePhaseTransition(data);
        }).catch(function (err) {
          // If write_article isn't reachable from current phase, try jumping through phases
          console.log('[DEV] Direct skip failed, trying phase sequence...');
          // Try chaining: briefing -> read_b -> read_a -> investigate needs findings...
          // Simplest: just navigate to conclusions then write_article
          Api.submitChoice(sessionId, 'conclusions', 'DEV skip').then(function (d1) {
            return Api.submitChoice(sessionId, 'write_article', 'DEV skip');
          }).then(function (d2) {
            handlePhaseTransition(d2);
          }).catch(function (e2) {
            console.error('[DEV] Could not skip to write_article:', e2);
          });
        });
      });
    }

    // Keyboard: Escape dismisses error section, returns to welcome
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && state.section === 'error') {
        clearSession();
        updateState({ section: 'welcome', error: null });
      }
    });

    // Attempt session recovery from sessionStorage before first render
    recoverSession();

    // Initial render (recoverSession may have already triggered renders
    // via updateState, but this ensures welcome is shown if no recovery)
    render();
  }

  // --------------------------------------------------------------------------
  // Public API (cross-module interface)
  // --------------------------------------------------------------------------

  window.App = {
    updateState: updateState,
    getState: getState,
    render: render,
    renderPhase: renderPhase,
    loadFirstTask: loadFirstTask,
    handlePhaseTransition: handlePhaseTransition,
    startPostTaskFlow: startPostTaskFlow,
    renderDebrief: renderDebrief,
    renderReveal: renderReveal,
    skipCurrentTask: skipCurrentTask
  };

  // --------------------------------------------------------------------------
  // Bootstrap
  // --------------------------------------------------------------------------

  document.addEventListener('DOMContentLoaded', init);

})();
