/* ==========================================================================
   Makaronas — investigation.js
   Investigation tree UI: tree navigation, key finding tracking,
   sessionStorage persistence, submission via POST /choice.
   Phase 5c — where students become detectives.
   ========================================================================== */

(function () {
  'use strict';

  // --------------------------------------------------------------------------
  // Internal State (module-scoped, not in App state — same pattern as dialogue.js)
  // --------------------------------------------------------------------------

  var treeState = {
    phaseId: null,
    allBlocks: [],          // flat array of ALL SearchResultBlocks from content
    queryToBlock: null,     // Map<queryText, block> — built once on render
    expandedQueries: [],    // query strings the student has expanded
    markedFindings: [],     // block IDs the student has marked as key findings
    minKeyFindings: 2,
    submitTarget: null
  };

  // DOM references (set during render, cleared on cleanup)
  var treeContainer = null;
  var findingsList = null;
  var progressEl = null;
  var submitBtn = null;
  var contentPanel = null;
  var interactionPanel = null;
  var beforeUnloadHandler = null;

  // Storage key — same string as dialogue.js DRAFT_KEY and app.js STORAGE_KEYS.interactionState
  var STORAGE_KEY = 'makaronas_interaction_state';

  // --------------------------------------------------------------------------
  // Public API
  // --------------------------------------------------------------------------

  /**
   * Renders the investigation tree in the content panel and controls
   * in the interaction panel. Called from interactions.js dispatch.
   */
  function renderInvestigation(cPanel, iPanel, interaction, phaseData) {
    contentPanel = cPanel;
    interactionPanel = iPanel;

    // Initialize tree state
    treeState.phaseId = phaseData.current_phase;
    treeState.allBlocks = extractSearchResultBlocks(phaseData.content || []);
    treeState.queryToBlock = buildQueryIndex(treeState.allBlocks);
    treeState.expandedQueries = [];
    treeState.markedFindings = [];
    treeState.minKeyFindings = interaction.min_key_findings || 2;
    treeState.submitTarget = interaction.submit_target;

    // --- Content panel: restructure to show only starting query cards ---
    restructureContentPanel(interaction.starting_queries || []);

    // --- Interaction panel: show trickster message + compact controls ---
    // Render trickster message above controls (it gets cleared by renderInteraction)
    if (phaseData.trickster_intro) {
      var tricksterMsg = document.createElement('div');
      tricksterMsg.className = 'investigation-trickster-message trickster-response';
      if (window.Renderer) {
        tricksterMsg.innerHTML = window.Renderer.renderMarkdown(phaseData.trickster_intro);
      } else {
        tricksterMsg.textContent = phaseData.trickster_intro;
      }
      interactionPanel.appendChild(tricksterMsg);
    }

    renderControls();

    // --- beforeunload persistence ---
    if (beforeUnloadHandler) {
      window.removeEventListener('beforeunload', beforeUnloadHandler);
    }
    beforeUnloadHandler = function () {
      persistState();
    };
    window.addEventListener('beforeunload', beforeUnloadHandler);
  }

  /**
   * Clears investigation state. Called from renderPhase() on phase transitions.
   */
  function clearInvestigation() {
    treeState.phaseId = null;
    treeState.allBlocks = [];
    treeState.queryToBlock = null;
    treeState.expandedQueries = [];
    treeState.markedFindings = [];
    treeState.minKeyFindings = 2;
    treeState.submitTarget = null;

    if (beforeUnloadHandler) {
      window.removeEventListener('beforeunload', beforeUnloadHandler);
      beforeUnloadHandler = null;
    }

    treeContainer = null;
    findingsList = null;
    progressEl = null;
    submitBtn = null;
    contentPanel = null;
    interactionPanel = null;
  }

  /**
   * Recovers investigation state from sessionStorage after page refresh.
   * Called from app.js recoverSession() AFTER renderPhase() has already called
   * renderInvestigation() via the interactions.js dispatch. This function only
   * replays expansions and marks on the already-rendered tree.
   */
  function recoverInvestigation(cPanel, iPanel, phaseData) {
    var stored = loadPersistedState();
    if (!stored || stored.phase_id !== phaseData.current_phase) {
      // Stale or no stored state — nothing to recover
      return;
    }

    // Tree is already rendered by renderPhase() → renderInteraction() → renderInvestigation()
    // Just replay the expansions and marks

    // Rehydrate expanded queries (top-down order so parent containers exist)
    var queriesToExpand = stored.expanded_queries || [];
    for (var i = 0; i < queriesToExpand.length; i++) {
      expandNodeByQuery(queriesToExpand[i]);
    }

    // Rehydrate marked findings
    var findingsToMark = stored.marked_findings || [];
    for (var j = 0; j < findingsToMark.length; j++) {
      markFindingById(findingsToMark[j]);
    }
  }

  // --------------------------------------------------------------------------
  // Content Panel: Tree Restructuring
  // --------------------------------------------------------------------------

  /**
   * Restructures the content panel to show a tree of starting query blocks.
   * renderPhase() has already rendered ALL blocks flat. We hide all search result
   * blocks, create a tree container, and move starting query blocks into it.
   */
  function restructureContentPanel(startingQueries) {
    if (!contentPanel) return;

    // Hide ALL search result blocks (they're rendered flat by renderPhase)
    var allSearchResults = contentPanel.querySelectorAll('.block-search-result');
    for (var i = 0; i < allSearchResults.length; i++) {
      allSearchResults[i].style.display = 'none';
    }

    // Create tree container
    treeContainer = document.createElement('div');
    treeContainer.className = 'investigation-tree';
    treeContainer.setAttribute('role', 'tree');
    treeContainer.setAttribute('aria-label', (window.I18n && window.I18n.investigation_heading) || 'Investigation');

    // Move starting query blocks into the tree container
    for (var j = 0; j < startingQueries.length; j++) {
      var queryText = startingQueries[j];
      var block = treeState.queryToBlock ? treeState.queryToBlock.get(queryText) : null;
      if (!block) continue;

      var el = document.getElementById('block-' + block.id);
      if (!el) continue;

      el.style.display = '';
      treeContainer.appendChild(el);

      // Add mark button to key finding cards
      addMarkButtonIfNeeded(el, block);
    }

    // Insert tree at the TOP of content panel (before articles, not after)
    contentPanel.insertBefore(treeContainer, contentPanel.firstChild);

    // Delegated click handler on tree container
    treeContainer.addEventListener('click', handleTreeClick);
  }

  // --------------------------------------------------------------------------
  // Expansion Logic
  // --------------------------------------------------------------------------

  /**
   * Handles clicks on the tree container. Dispatches between mark button
   * clicks and expand/collapse clicks.
   */
  function handleTreeClick(e) {
    // Check if locked (API submission in progress)
    var appState = window.App.getState();
    if (appState.locked) return;

    // Priority 1: Mark button click
    var markBtn = e.target.closest('.investigation-mark-btn');
    if (markBtn) {
      e.stopPropagation();
      toggleFinding(markBtn);
      return;
    }

    // Priority 2: Expand/collapse on search result with children
    // aria-expanded is on the card element; data-child-queries is on the __children container
    var resultCard = e.target.closest('.block-search-result');
    if (resultCard && resultCard.hasAttribute('aria-expanded')) {
      toggleExpand(resultCard);
    }
  }

  /**
   * Toggles expand/collapse on a search result card.
   */
  function toggleExpand(cardEl) {
    var childrenContainer = cardEl.querySelector(':scope > .block-search-result__children');
    if (!childrenContainer) return;

    var isExpanded = cardEl.getAttribute('aria-expanded') === 'true';

    if (isExpanded) {
      // Collapse
      childrenContainer.hidden = true;
      cardEl.setAttribute('aria-expanded', 'false');

      // Remove query from expandedQueries
      var queryEl = cardEl.querySelector(':scope > .block-search-result__query');
      var queryText = queryEl ? queryEl.textContent : null;
      if (queryText) {
        var idx = treeState.expandedQueries.indexOf(queryText);
        if (idx !== -1) treeState.expandedQueries.splice(idx, 1);
      }
    } else {
      // Expand
      expandCard(cardEl, childrenContainer);
    }

    persistState();
  }

  /**
   * Expands a card by resolving child queries and moving child elements
   * into the children container.
   */
  function expandCard(cardEl, childrenContainer) {
    var childQueriesJson = childrenContainer.getAttribute('data-child-queries');
    if (!childQueriesJson) return;

    var childQueries;
    try {
      childQueries = JSON.parse(childQueriesJson);
    } catch (e) {
      return;
    }

    // Only populate if not already populated
    if (childrenContainer.children.length === 0) {
      for (var i = 0; i < childQueries.length; i++) {
        var childQueryText = childQueries[i];
        var childBlock = treeState.queryToBlock ? treeState.queryToBlock.get(childQueryText) : null;
        if (!childBlock) continue; // Silently skip missing blocks (§8.1)

        var childEl = document.getElementById('block-' + childBlock.id);
        if (!childEl) continue;

        childEl.style.display = '';
        childrenContainer.appendChild(childEl);

        // Add mark button to key finding children
        addMarkButtonIfNeeded(childEl, childBlock);
      }
    }

    childrenContainer.hidden = false;
    cardEl.setAttribute('aria-expanded', 'true');

    // Track expanded query
    var queryEl = cardEl.querySelector(':scope > .block-search-result__query');
    var queryText = queryEl ? queryEl.textContent : null;
    if (queryText && treeState.expandedQueries.indexOf(queryText) === -1) {
      treeState.expandedQueries.push(queryText);
    }
  }

  /**
   * Expands a node by its query text — used during recovery to replay expansions.
   */
  function expandNodeByQuery(queryText) {
    var block = treeState.queryToBlock ? treeState.queryToBlock.get(queryText) : null;
    if (!block) return;

    var cardEl = document.getElementById('block-' + block.id);
    if (!cardEl) return;

    var childrenContainer = cardEl.querySelector(':scope > .block-search-result__children');
    if (!childrenContainer) return;

    expandCard(cardEl, childrenContainer);
    persistState();
  }

  // --------------------------------------------------------------------------
  // Key Finding Toggle
  // --------------------------------------------------------------------------

  /**
   * Adds a mark button inside a search result card if it's a key finding.
   */
  function addMarkButtonIfNeeded(el, block) {
    if (!block.is_key_finding) return;
    // Don't add duplicate buttons
    if (el.querySelector('.investigation-mark-btn')) return;

    var btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'btn investigation-mark-btn';
    btn.dataset.blockId = block.id;
    btn.textContent = (window.I18n && window.I18n.investigation_mark) || 'Mark as finding';
    btn.setAttribute('aria-pressed', 'false');

    // Insert after the snippet or URL, before children container
    var childrenDiv = el.querySelector(':scope > .block-search-result__children');
    if (childrenDiv) {
      el.insertBefore(btn, childrenDiv);
    } else {
      el.appendChild(btn);
    }
  }

  /**
   * Toggles a finding's marked state.
   */
  function toggleFinding(markBtn) {
    var blockId = markBtn.dataset.blockId;
    if (!blockId) return;

    var idx = treeState.markedFindings.indexOf(blockId);
    var cardEl = markBtn.closest('.block-search-result');

    if (idx !== -1) {
      // Unmark
      treeState.markedFindings.splice(idx, 1);
      markBtn.textContent = (window.I18n && window.I18n.investigation_mark) || 'Mark as finding';
      markBtn.setAttribute('aria-pressed', 'false');
      markBtn.classList.remove('investigation-mark-btn--marked');
      if (cardEl) cardEl.classList.remove('block-search-result--marked');
    } else {
      // Mark
      treeState.markedFindings.push(blockId);
      markBtn.textContent = (window.I18n && window.I18n.investigation_marked) || 'Marked';
      markBtn.setAttribute('aria-pressed', 'true');
      markBtn.classList.add('investigation-mark-btn--marked');
      if (cardEl) cardEl.classList.add('block-search-result--marked');

      // Auto-expand children of marked findings (so student sees deeper trail)
      if (cardEl && cardEl.getAttribute('aria-expanded') === 'false') {
        toggleExpand(cardEl);
      }
    }

    updateControls();
    persistState();
  }

  /**
   * Marks a finding by block ID — used during recovery.
   */
  function markFindingById(blockId) {
    var el = document.getElementById('block-' + blockId);
    if (!el) return;

    var markBtn = el.querySelector('.investigation-mark-btn');
    if (!markBtn) return;

    // Only mark if not already marked
    if (treeState.markedFindings.indexOf(blockId) === -1) {
      toggleFinding(markBtn);
    }
  }

  // --------------------------------------------------------------------------
  // Interaction Panel Controls
  // --------------------------------------------------------------------------

  /**
   * Renders the investigation controls in the interaction panel:
   * findings summary, progress indicator, submit button.
   */
  function renderControls() {
    if (!interactionPanel) return;

    var controls = document.createElement('div');
    controls.className = 'investigation-controls';

    // Findings summary list
    var findingsSection = document.createElement('div');
    findingsSection.className = 'investigation-findings';

    var findingsLabel = document.createElement('h3');
    findingsLabel.className = 'investigation-findings__label';
    findingsLabel.textContent = (window.I18n && window.I18n.investigation_heading) || 'Investigation';
    findingsSection.appendChild(findingsLabel);

    findingsList = document.createElement('ul');
    findingsList.className = 'investigation-findings-list';
    findingsSection.appendChild(findingsList);

    var emptyMsg = document.createElement('p');
    emptyMsg.className = 'investigation-findings-empty';
    emptyMsg.textContent = (window.I18n && window.I18n.investigation_no_findings) || 'No key findings yet';
    findingsSection.appendChild(emptyMsg);

    controls.appendChild(findingsSection);

    // Progress indicator
    progressEl = document.createElement('div');
    progressEl.className = 'investigation-progress';
    progressEl.setAttribute('aria-live', 'polite');
    controls.appendChild(progressEl);

    // Submit button
    submitBtn = document.createElement('button');
    submitBtn.type = 'button';
    submitBtn.className = 'btn investigation-submit-btn';
    submitBtn.textContent = (window.I18n && window.I18n.investigation_submit) || 'Submit findings';
    submitBtn.disabled = true;
    submitBtn.title = (window.I18n && window.I18n.investigation_submit_disabled) || 'Find more key findings';
    submitBtn.addEventListener('click', handleSubmit);
    controls.appendChild(submitBtn);

    interactionPanel.appendChild(controls);

    updateControls();
  }

  /**
   * Updates the findings list, progress indicator, and submit button state.
   */
  function updateControls() {
    if (!findingsList || !progressEl || !submitBtn) return;

    var found = treeState.markedFindings.length;
    var required = treeState.minKeyFindings;

    // Update progress text
    var progressLabel = (window.I18n && window.I18n.investigation_progress) || 'Findings found';
    progressEl.textContent = progressLabel + ': ' + found + ' / ' + required;

    // Update findings list
    findingsList.innerHTML = '';
    var emptyMsg = interactionPanel ? interactionPanel.querySelector('.investigation-findings-empty') : null;

    if (found === 0) {
      if (emptyMsg) emptyMsg.style.display = '';
    } else {
      if (emptyMsg) emptyMsg.style.display = 'none';
      for (var i = 0; i < treeState.markedFindings.length; i++) {
        var blockId = treeState.markedFindings[i];
        var block = findBlockById(blockId);
        var li = document.createElement('li');
        li.className = 'investigation-findings-list__item';
        li.textContent = block ? block.title : blockId;
        findingsList.appendChild(li);
      }
    }

    // Update submit button
    if (found >= required) {
      submitBtn.disabled = false;
      submitBtn.title = '';
    } else {
      submitBtn.disabled = true;
      submitBtn.title = (window.I18n && window.I18n.investigation_submit_disabled) || 'Find more key findings';
    }
  }

  /**
   * Handles submit button click — sends findings via POST /choice.
   */
  function handleSubmit() {
    if (!treeState.submitTarget) return;

    var appState = window.App.getState();
    var sessionId = appState.session && appState.session.session_id;
    if (!sessionId) return;

    // Immediate visual feedback — disable button (double-protection pattern from 5a)
    submitBtn.disabled = true;

    Api.submitChoice(sessionId, treeState.submitTarget, null).then(function (data) {
      // Clear persisted state on successful submit
      clearPersistedState();
      window.App.handlePhaseTransition(data);
    }).catch(function (err) {
      // Re-enable submit if it still meets threshold
      if (treeState.markedFindings.length >= treeState.minKeyFindings) {
        submitBtn.disabled = false;
      }
      window.App.updateState({ error: { message: err.message } });
    });
  }

  // --------------------------------------------------------------------------
  // Session Storage Persistence
  // --------------------------------------------------------------------------

  /**
   * Persists current investigation state to sessionStorage.
   */
  function persistState() {
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify({
        type: 'investigation',
        phase_id: treeState.phaseId,
        expanded_queries: treeState.expandedQueries,
        marked_findings: treeState.markedFindings
      }));
    } catch (e) {
      // sessionStorage unavailable or full — silently fail
    }
  }

  /**
   * Loads persisted investigation state from sessionStorage.
   */
  function loadPersistedState() {
    try {
      var stored = sessionStorage.getItem(STORAGE_KEY);
      if (!stored) return null;
      var parsed = JSON.parse(stored);
      // Check type — distinguish from dialogue drafts
      if (parsed && (parsed.type === 'investigation' || parsed.expanded_queries)) {
        return parsed;
      }
      return null;
    } catch (e) {
      return null;
    }
  }

  /**
   * Clears persisted investigation state.
   */
  function clearPersistedState() {
    try {
      sessionStorage.removeItem(STORAGE_KEY);
    } catch (e) {
      // Ignore
    }
  }

  // --------------------------------------------------------------------------
  // Helpers
  // --------------------------------------------------------------------------

  /**
   * Extracts search_result blocks from the content array.
   */
  function extractSearchResultBlocks(content) {
    var blocks = [];
    for (var i = 0; i < content.length; i++) {
      if (content[i].type === 'search_result') {
        blocks.push(content[i]);
      }
    }
    return blocks;
  }

  /**
   * Builds a Map<queryText, block> index for O(1) child resolution.
   * First match wins if duplicates exist (§8.8).
   */
  function buildQueryIndex(blocks) {
    var index = new Map();
    for (var i = 0; i < blocks.length; i++) {
      var block = blocks[i];
      if (block.query && !index.has(block.query)) {
        index.set(block.query, block);
      }
    }
    return index;
  }

  /**
   * Finds a block by its ID from the allBlocks array.
   */
  function findBlockById(blockId) {
    for (var i = 0; i < treeState.allBlocks.length; i++) {
      if (treeState.allBlocks[i].id === blockId) {
        return treeState.allBlocks[i];
      }
    }
    return null;
  }

  // --------------------------------------------------------------------------
  // Public API
  // --------------------------------------------------------------------------

  window.Investigation = {
    renderInvestigation: renderInvestigation,
    clearInvestigation: clearInvestigation,
    recoverInvestigation: recoverInvestigation
  };

})();
