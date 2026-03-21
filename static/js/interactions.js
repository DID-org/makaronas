/* ==========================================================================
   Makaronas — interactions.js
   Interaction dispatcher: renders controls in the interaction panel based on
   the phase's interaction config. Phase 5a handles "button" type.
   Phase 5b adds "freeform", Phase 5c adds "investigation".
   ========================================================================== */

(function () {
  'use strict';

  /**
   * Renders the appropriate interaction controls into the interaction panel.
   * Dispatches on phaseData.interaction.type. Clears the panel first.
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
