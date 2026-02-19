# Pre-Built Roadmaps

This directory holds **pre-built roadmap** files — curated task sequences that teachers can assign to their classes without customisation.

## What's a Roadmap?

A roadmap is an ordered sequence of tasks that forms a learning path. Pre-built roadmaps are authored by the content team and ship with the platform. Teachers can also create custom roadmaps via the Composer or the API (`POST /api/v1/teacher/roadmaps`).

## Directory Structure

```
content/roadmaps/prebuilt/
├── README.md                          ← you are here
├── introduction_to_manipulation.json  ← example: starter sequence
├── advanced_social_engineering.json   ← example: deeper sequence
└── full_curriculum_10_weeks.json      ← example: complete course
```

**No roadmap files exist yet.** They'll be authored when the task content is created (V2+).

## Expected Format

The roadmap schema is defined in V2 (Task Engine). Below is a preview — **subject to change**.

```json
{
  "roadmap_id": "intro_manipulation_v1",
  "title": "Introduction to Manipulation",
  "description": "A 5-task sequence covering basic manipulation techniques",
  "task_ids": [
    "001_clickbait_trap",
    "002_misleading_stats",
    "003_social_pressure",
    "004_appeal_to_authority",
    "005_false_urgency"
  ],
  "estimated_minutes": 50,
  "difficulty_range": [1, 3],
  "notes": "Recommended as a starting point for new classes."
}
```

## How They're Used

- Teachers browse pre-built roadmaps via `GET /api/v1/teacher/roadmaps`
- Students start a session with an optional `roadmap_id` parameter (`POST /api/v1/student/session`)
- The session engine (V2) uses the roadmap to determine task order
