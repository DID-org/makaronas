/* ==========================================================================
   Makaronas — i18n.js
   Lithuanian UI strings. Flat map, \u escapes for diacritics.
   When V12 adds localization, this file becomes one locale in a loader.
   ========================================================================== */

window.I18n = {

  // ---------------------------------------------------------------------------
  // Welcome
  // ---------------------------------------------------------------------------
  welcome_title: 'Makaronas',
  welcome_description: 'Mokymo platforma, kurioje dirbtinis intelektas padeda atpa\u017Einti informacin\u0119 manipuliacij\u0105.',
  btn_start: 'Prad\u0117ti',

  // ---------------------------------------------------------------------------
  // Task
  // ---------------------------------------------------------------------------
  section_task: 'U\u017Eduotis',
  content_heading: 'Turinys',
  interaction_heading: 'S\u0105veika',

  // ---------------------------------------------------------------------------
  // Dialogue (Phase 5b+)
  // ---------------------------------------------------------------------------
  btn_send: 'Si\u0173sti',
  placeholder_message: '\u012Eveskite savo atsakym\u0105\u2026',
  dialogue_typing: 'Tricksteris ra\u0161o\u2026',
  dialogue_send_disabled: 'Palaukite atsakymo',
  dialogue_error_retry: 'Bandyti i\u0161 naujo',
  dialogue_error_partial: 'Atsakymas nutr\u016Bko. Bandykite dar kart\u0105.',

  // ---------------------------------------------------------------------------
  // Debrief / Reveal
  // ---------------------------------------------------------------------------
  debrief_heading: 'Aptarimas',
  reveal_heading: 'I\u0161vados',
  btn_next_task: 'Kitas u\u017Edavinys',

  // ---------------------------------------------------------------------------
  // Session End
  // ---------------------------------------------------------------------------
  session_end_title: 'Sesija baigta',
  session_end_message: 'A\u010Di\u016B, kad dalyvavai.',

  // ---------------------------------------------------------------------------
  // Error
  // ---------------------------------------------------------------------------
  error_heading: 'Klaida',
  btn_retry: 'Bandyti dar kart\u0105',
  error_generic: '\u012Evyko klaida. Bandykite dar kart\u0105.',
  error_session_expired: 'Sesija pasibaig\u0117. Prad\u0117kite i\u0161 naujo.',
  error_task_not_found: 'U\u017Eduotis nerasta.',
  error_no_task: 'N\u0117ra priskirtos u\u017Eduoties.',
  error_no_phase: 'N\u0117ra aktyvaus etapo.',
  error_ai_unavailable: 'AI \u0161iuo metu nepasiekiamas. Bandykite v\u0117liau.',
  error_ai_timeout: 'AI atsakymas u\u017Etruko per ilgai. Bandykite dar kart\u0105.',
  error_rate_limit: 'Palauk truput\u012F.',
  error_network: 'Tinklo klaida. Patikrinkite interneto ry\u0161\u012F.',
  error_unauthorized: 'Prisijungimas nebegaliojas. Prad\u0117kite i\u0161 naujo.',
  error_asset: 'Turinys nepasiekiamas',

  // ---------------------------------------------------------------------------
  // Generation (Phase 7c)
  // ---------------------------------------------------------------------------
  btn_submit_generation: 'Si\u0173sti Tricksteriui',
  placeholder_generation: 'Para\u0161ykite savo manipuliacijos bandym\u0105\u2026',

  // ---------------------------------------------------------------------------
  // Interaction (Phase 5a)
  // ---------------------------------------------------------------------------
  interaction_unsupported: '\u0160is s\u0105veikos tipas dar nepalaikomas',
  task_load_error: 'Nepavyko u\u017Ekrauti u\u017Eduoties',

  // ---------------------------------------------------------------------------
  // Loading / Status
  // ---------------------------------------------------------------------------
  loading: 'Kraunama\u2026',
  ai_thinking: 'AI galvoja\u2026',

  // ---------------------------------------------------------------------------
  // Accessibility
  // ---------------------------------------------------------------------------
  skip_to_content: 'Pereiti prie turinio',
  search_key_finding: 'Svarbus radinys',
  search_dead_end: 'Aklavi\u0117t\u0117'

};
