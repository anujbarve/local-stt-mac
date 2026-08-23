# Second Brain — Complete Architecture Plan

## Vision

A fully local, voice-driven personal knowledge system that encompasses every aspect of human life. Not a work tool. Not a personal organizer. A complete external memory that understands intentions, actions, priorities, relationships, and patterns across every domain of your existence.

**Core principles:**
- Fully local. No cloud. No API. No data leaves the machine.
- Voice-first input via voiceassist pipeline.
- Neural network handles 85% of daily tasks. LLM handles deep reasoning.
- Improves continuously through use and corrections.
- Plain markdown vault. Portable. Version-controllable. Obsidian-compatible.

---

## Part 1: The Domain Map

Every piece of information a human generates falls into one of these domains. The system must handle ALL of them.

### Domain 1: People & Relationships

Every person you interact with, and every relationship you have.

**Entity: Person**

| Field | Type | Example |
|---|---|---|
| name | string | "Sarah Chen" |
| nickname | string | "Sar" |
| role | string | "colleague", "friend", "doctor" |
| relationship | string | "coworker", "family", "mentor", "client" |
| organization | string | "Acme Corp" |
| birthday | date | "1992-03-15" |
| anniversary | date | "2020-06-20" |
| email | string | "sarah@example.com" |
| phone | string | "+1-555-0123" |
| location | string | "San Francisco" |
| how_we_met | string | "College roommate" |
| notes | text | "Prefers email over text" |
| communication_style | string | "formal", "casual", "humorous" |
| topics_of_interest | list | ["AI", "hiking", "cooking"] |
| last_contact | date | "2026-08-20" |
| contact_frequency | string | "weekly", "monthly", "rarely" |
| social_profiles | dict | {twitter: "@sarahc", linkedin: "..."} |
| preferences | dict | {food: "vegetarian", coffee: "oat milk latte"} |
| important_dates | list | [{event: "birthday", date: "03-15"}, ...] |
| projects_shared | list | ["voiceassist", "marketing Q3"] |
| sentiment | string | "positive", "neutral", "complicated" |

**Query patterns:**
- "When is Sarah's birthday?" → person.birthday
- "Who have I been working with on project X?" → person.projects_shared
- "What did I discuss with Mike last time?" → conversation_history[person=Mike]
- "Who should I reach out to about AI?" → person.topics_of_interest contains "AI"
- "Who haven't I contacted in a while?" → ORDER BY last_contact ASC
- "What does Sarah like?" → person.preferences
- "How do I know this person?" → person.how_we_met

---

### Domain 2: Projects

Every project you're involved in, at any scale.

**Entity: Project**

| Field | Type | Example |
|---|---|---|
| name | string | "Voice Assistant" |
| description | text | "Fully local speech-to-text CLI" |
| status | enum | planning, active, paused, completed, abandoned |
| priority | enum | critical, high, medium, low |
| category | string | "work", "personal", "research", "side-hustle", "learning" |
| start_date | date | "2026-06-01" |
| target_date | date | "2026-09-30" |
| completion_date | date | null |
| progress | float | 0.0 to 1.0 (0% to 100%) |
| people | list | ["Sarah", "Mike"] |
| technologies | list | ["Python", "PyTorch", "ONNX"] |
| dependencies | list | ["project-A", "project-B"] |
| blockers | list | ["Need GPU access", "Waiting on API approval"] |
| milestones | list | [{name: "MVP", date: "2026-08-01", done: true}] |
| decisions | list | ["Use MLX over ONNX for Whisper", ...] |
| tags | list | ["ai", "voice", "python"] |
| repository | string | "github.com/user/voiceassist" |
| notes | text | "Main priority this quarter" |

**Query patterns:**
- "Which projects are pending?" → project.status == active
- "What's the status of project X?" → project WHERE name LIKE X
- "What kind of projects am I working on?" → GROUP BY project.category
- "What projects involve Python?" → project.technologies contains "Python"
- "What are my highest priority projects?" → ORDER BY priority
- "What projects am I doing for work vs personal?" → GROUP BY category
- "What projects are blocked?" → project.blockers IS NOT EMPTY
- "What projects did I complete this year?" → project.status == completed AND year
- "What's the overall progress across all projects?" → AVG(progress)

---

### Domain 3: Tasks & Actions

Every actionable item, from "buy milk" to "deploy the ML pipeline."

**Entity: Task**

| Field | Type | Example |
|---|---|---|
| title | string | "Fix VAD threshold bug" |
| description | text | "The threshold is too sensitive, causing false positives" |
| status | enum | todo, in_progress, done, blocked, cancelled |
| priority | enum | critical, high, medium, low, someday |
| urgency | enum | immediate, soon, later, someday |
| due_date | date | "2026-08-25" |
| due_time | time | "17:00" |
| project | string | "voiceassist" |
| assignee | string | "me" |
| context | string | "office", "home", "phone", "computer" |
| energy_level | string | "high", "medium", "low" |
| estimated_duration | duration | "30min", "2hr" |
| dependencies | list | ["task-X must be done first"] |
| tags | list | ["bug", "vad", "urgent"] |
| source | string | "meeting with Mike", "morning thought" |
| recurrence | string | "daily", "weekly", "monthly", null |
| completed_at | datetime | null |
| notes | text | "Check the Silero VAD docs first" |

**Query patterns:**
- "What tasks are overdue?" → task.due_date < today AND status != done
- "What should I focus on today?" → ORDER BY urgency, priority (filtered by context)
- "What's blocking the deployment?" → task.status == blocked
- "What did I accomplish this week?" → task.status == done AND completed_at this_week
- "What tasks are low energy?" → task.energy_level == low (for when tired)
- "What tasks can I do from my phone?" → task.context == phone
- "What tasks are related to project X?" → task.project == X
- "What tasks am I avoiding?" → task.status == todo AND due_date < today
- "How many tasks did I complete this month?" → COUNT(done) this_month

---

### Domain 4: Ideas & Thoughts

Every idea, no matter how raw or half-baked.

**Entity: Idea**

| Field | Type | Example |
|---|---|---|
| title | string | "Circular buffer for pre-roll" |
| content | text | "Use a ring buffer to capture audio before trigger..." |
| stage | enum | fleeting, developing, mature, implemented, abandoned |
| origin | string | "morning walk", "reading PKM article", "shower thought" |
| domain | string | "technical", "creative", "business", "personal" |
| related_projects | list | ["voiceassist"] |
| related_ideas | list | ["idea-X", "idea-Y"] |
| potential_impact | string | "high", "medium", "low" |
| effort_to_implement | string | "small", "medium", "large" |
| tags | list | ["architecture", "audio", "optimization"] |
| first_captured | datetime | "2026-08-23T07:15:00Z" |
| last_reviewed | datetime | "2026-08-23T07:15:00Z" |
| review_count | int | 0 |

**Query patterns:**
- "What ideas do I have about X?" → idea.tags/topics MATCH X
- "How does this idea connect to previous ones?" → idea.related_ideas + embedding similarity
- "What ideas came from that podcast?" → idea.origin == "podcast"
- "What are my most promising ideas?" → ORDER BY potential_impact
- "What ideas have I not reviewed in a while?" → ORDER BY last_reviewed
- "What ideas relate to project X?" → idea.related_projects contains X
- "Show me my fleeting ideas" → idea.stage == fleeting

---

### Domain 5: Research & Learning

Everything you've read, watched, studied, or learned.

**Entity: Learning**

| Field | Type | Example |
|---|---|---|
| topic | string | "Zettelkasten method" |
| source_type | enum | article, book, video, podcast, conversation, course, paper |
| source_title | string | "How to Take Smart Notes" |
| source_author | string | "Sönke Ahrens" |
| source_url | string | "https://..." |
| key_insights | list | ["Atomic notes enable compounding", ...] |
| my_summary | text | "One idea per note, link densely..." |
| my_opinion | text | "I think this is overengineered for most people" |
| confidence | string | "high", "medium", "low", "unverified" |
| related_topics | list | ["PKM", "note-taking", "knowledge-management"] |
| related_projects | list | ["second-brain"] |
| date_consumed | date | "2026-08-20" |
| quality_rating | int | 1-5 stars |
| would_recommend | bool | true |
| open_questions | list | ["How does this scale to 10K+ notes?"] |
| tags | list | ["productivity", "pkm", "zettelkasten"] |

**Query patterns:**
- "What did I learn about neural networks?" → learning.topic MATCH
- "Summarize what I know about X" → AGGREGATE learning WHERE topic MATCH X
- "What articles did I read about PKM?" → learning.source_type == article AND topic MATCH
- "What's the best resource on X?" → learning.quality_rating DESC
- "What have I learned this month?" → learning.date_consumed this_month
- "What topics am I researching?" → DISTINCT learning.topic
- "What books have I read?" → learning.source_type == book
- "What do I think about X?" → learning.my_opinion WHERE topic MATCH

---

### Domain 6: Decisions & Rationale

Every decision you've made, and WHY you made it.

**Entity: Decision**

| Field | Type | Example |
|---|---|---|
| title | string | "Use MLX over ONNX for Whisper" |
| decision | string | "MLX" |
| rationale | text | "MLX is optimized for Apple Silicon, 2x faster inference" |
| alternatives | list | ["ONNX Runtime", "ctransformers", "whisper.cpp"] |
| tradeoffs | text | "MLX is macOS-only but we're Apple-only anyway" |
| reversible | bool | true |
| confidence | string | "high", "medium", "low" |
| made_on | date | "2026-07-15" |
| made_by | string | "me" |
| project | string | "voiceassist" |
| context | text | "After benchmarking all options on M2 MacBook" |
| revisit_date | date | "2026-12-01" (if reversible) |
| outcome | string | null (filled in later: "good", "bad", "neutral") |
| tags | list | ["architecture", "ml", "apple-silicon"] |

**Query patterns:**
- "Why did I choose X over Y?" → decision WHERE alternatives CONTAINS Y
- "When did I decide to drop the cloud backend?" → decision WHERE title MATCH
- "What decisions did I make this quarter?" → decision.made_on this_quarter
- "What decisions need revisiting?" → decision.revisit_date <= today
- "What decisions turned out badly?" → decision.outcome == bad
- "What were the tradeoffs for X?" → decision.tradeoffs WHERE title MATCH

---

### Domain 7: Meetings & Conversations

Every meeting, call, and important conversation.

**Entity: Meeting**

| Field | Type | Example |
|---|---|---|
| title | string | "Sprint planning" |
| date | datetime | "2026-08-23T10:00:00Z" |
| duration | duration | "45min" |
| participants | list | ["Sarah", "Mike", "John"] |
| type | enum | meeting, call, coffee, chat, interview |
| project | string | "voiceassist" |
| agenda | list | ["Review blockers", "Plan next sprint"] |
| discussion | text | "Discussed the VAD threshold issue..." |
| decisions_made | list | ["decision-1", "decision-2"] |
| action_items | list | ["Fix VAD bug", "Update README"] |
| follow_ups | list | [{who: "Mike", what: "Send API docs", when: "Friday"}] |
| my_notes | text | "Sarah seems stressed about Q3 deadline" |
| sentiment | string | "productive", "tense", "casual", "formal" |
| location | string | "Conference room B", "Zoom", "Coffee shop" |
| tags | list | ["sprint", "planning", "team"] |

**Query patterns:**
- "What did I discuss with X last time?" → meeting WHERE participants CONTAINS X, ORDER BY date DESC
- "What meetings did I have this week?" → meeting.date this_week
- "What action items came from the sprint planning?" → meeting.action_items WHERE title MATCH
- "Who was in the meeting about X?" → meeting.participants WHERE project/discussion MATCH X
- "What follow-ups do I have?" → meeting.follow_ups WHERE when <= today

---

### Domain 8: Daily Journal & Observations

Everything that happens day-to-day.

**Entity: DailyLog**

| Field | Type | Example |
|---|---|---|
| date | date | "2026-08-23" |
| entries | list | [{time, content, type, mood}] |
| mood | string | "good", "neutral", "stressed", "energetic", "tired" |
| energy_level | string | "high", "medium", "low" |
| weather | string | "sunny", "rainy" |
| location | string | "office", "home", "traveling" |
| highlights | list | ["Shipped the VAD fix", "Great meeting with Sarah"] |
| frustrations | list | ["Git merge conflict took 2 hours"] |
| gratitude | list | ["Helpful code review from Mike"] |
| exercise | string | "30min run" |
| sleep_hours | float | 7.5 |
| meals | list | ["oatmeal", "salad", "pasta"] |

**Query patterns:**
- "What did I do last Tuesday?" → daily_log WHERE date == Tuesday
- "What happened this month?" → AGGREGATE daily_log this_month
- "How's my mood been lately?" → daily_log.mood last_7_days
- "What were my highlights this week?" → daily_log.highlights this_week
- "What's my average sleep this month?" → AVG(sleep_hours) this_month
- "When was I most productive?" → daily_log WHERE energy_level == high

---

### Domain 9: Reminders & Time-Sensitive Items

Anything with a temporal component.

**Entity: Reminder**

| Field | Type | Example |
|---|---|---|
| title | string | "Call dentist" |
| reminder_date | datetime | "2026-08-25T09:00:00Z" |
| recurrence | string | "none", "daily", "weekly", "monthly", "yearly" |
| priority | enum | critical, high, medium, low |
| category | string | "health", "finance", "personal", "work" |
| related_entity | string | "Dentist: Dr. Smith" |
| completed | bool | false |
| snooze_count | int | 0 |
| last_snoozed | datetime | null |
| notes | text | "Bring insurance card" |

**Query patterns:**
- "What do I need to remember today?" → reminder WHERE reminder_date == today
- "What reminders are coming up?" → reminder WHERE reminder_date > today, ORDER BY date
- "What recurring reminders do I have?" → reminder WHERE recurrence != none
- "What health reminders do I have?" → reminder WHERE category == health

---

### Domain 10: Preferences & Tastes

Everything you like, dislike, and prefer.

**Entity: Preference**

| Field | Type | Example |
|---|---|---|
| category | string | "food", "music", "tools", "clothing", "travel" |
| item | string | "dark roast coffee" |
| sentiment | enum | love, like, neutral, dislike, hate |
| context | string | "morning", "work", "casual" |
| notes | text | "Oat milk, no sugar" |
| first_noted | date | "2026-01-15" |
| confidence | string | "confirmed", "tentative" |

**Query patterns:**
- "What's my coffee order?" → preference WHERE item MATCH "coffee"
- "What tools do I prefer?" → preference WHERE category == tools
- "What do I like about Obsidian?" → preference WHERE item MATCH "Obsidian" AND sentiment positive
- "Do I prefer X or Y?" → preference WHERE item IN [X, Y]

---

### Domain 11: Health & Medical

Medical appointments, medications, health goals.

**Entity: HealthRecord**

| Field | Type | Example |
|---|---|---|
| type | enum | appointment, medication, symptom, goal, test_result |
| title | string | "Annual checkup" |
| date | date | "2026-09-15" |
| provider | string | "Dr. Smith, City Clinic" |
| details | text | "Blood work results: all normal" |
| follow_up | string | "Check cholesterol in 6 months" |
| medication | string | {name: "Vitamin D", dosage: "2000 IU", frequency: "daily"} |
| tags | list | ["checkup", "preventive"] |

**Query patterns:**
- "When is my next dentist appointment?" → health WHERE type == appointment AND provider MATCH
- "What medication am I on?" → health WHERE type == medication
- "When was my last checkup?" → health WHERE type == appointment, ORDER BY date DESC

---

### Domain 12: Finance & Subscriptions

Money-related records.

**Entity: FinanceRecord**

| Field | Type | Example |
|---|---|---|
| type | enum | subscription, purchase, goal, account |
| title | string | "Netflix subscription" |
| amount | float | 15.99 |
| currency | string | "USD" |
| frequency | string | "monthly", "yearly", "one-time" |
| next_payment | date | "2026-09-01" |
| category | string | "entertainment", "software", "utilities" |
| notes | text | "Family plan" |
| cancel_by | date | null |

**Query patterns:**
- "What subscriptions do I have?" → finance WHERE type == subscription
- "When does my Netflix renew?" → finance WHERE title MATCH "Netflix"
- "How much am I spending on software?" → SUM(amount) WHERE category == software

---

### Domain 13: Travel & Locations

Trips, places, travel preferences.

**Entity: TravelRecord**

| Field | Type | Example |
|---|---|---|
| type | enum | trip, place, preference, booking |
| title | string | "Tokyo trip" |
| destination | string | "Tokyo, Japan" |
| dates | range | "2026-11-01 to 2026-11-14" |
| purpose | string | "vacation" |
| bookings | list | [{type: "flight", details: "..."}] |
| preferences | list | ["love ramen", "prefer trains over taxis"] |
| notes | text | "Check visa requirements" |
| budget | float | 5000.00 |

**Query patterns:**
- "When is my next trip?" → travel WHERE type == trip AND dates > today
- "What do I know about Tokyo?" → travel WHERE destination MATCH
- "What's my travel preference for X?" → travel WHERE type == preference

---

### Domain 14: Goals & Plans

Short-term and long-term goals.

**Entity: Goal**

| Field | Type | Example |
|---|---|---|
| title | string | "Learn system design" |
| description | text | "Be able to design distributed systems from scratch" |
| timeframe | string | "Q3 2026", "2026", "5-year" |
| category | enum | career, personal, health, financial, learning, creative |
| progress | float | 0.0 to 1.0 |
| milestones | list | [{name: "Finish course", done: false}] |
| related_projects | list | ["system-design-study"] |
| status | enum | active, achieved, paused, abandoned |
| started | date | "2026-07-01" |
| target_date | date | "2026-12-31" |
| notes | text | "Focus on distributed databases first" |

**Query patterns:**
- "What are my goals for this quarter?" → goal WHERE timeframe MATCH current_quarter
- "How much progress on the book?" → goal WHERE title MATCH
- "What goals am I neglecting?" → goal.status == active AND progress < 0.3
- "What goals did I achieve this year?" → goal.status == achieved

---

### Domain 15: Lessons & Reflections

What you've learned from experience.

**Entity: Lesson**

| Field | Type | Example |
|---|---|---|
| title | string | "Test before refactoring" |
| content | text | "Refactored the audio module without tests, broke 3 things" |
| source_experience | string | "The great refactor disaster of August 2026" |
| principle | string | "Always have tests before refactoring" |
| domain | string | "engineering", "management", "personal" |
| confidence | string | "learned the hard way", "observed", "theoretical" |
| would_do_differently | text | "Write integration tests first, then refactor in small steps" |
| date_learned | date | "2026-08-20" |
| tags | list | ["testing", "refactoring", "best-practices"] |

**Query patterns:**
- "What did I learn from the failed deployment?" → lesson WHERE source_experience MATCH
- "What would I do differently next time?" → lesson.would_do_differently
- "What engineering lessons do I have?" → lesson WHERE domain == engineering
- "What are my core principles?" → lesson WHERE confidence == "learned the hard way"

---

### Domain 16: Tools & Technologies

Every tool, technology, and service you use.

**Entity: Tool**

| Field | Type | Example |
|---|---|---|
| name | string | "Obsidian" |
| category | string | "note-taking", "code-editor", "design" |
| url | string | "https://obsidian.md" |
| purpose | string | "Second brain / PKM" |
| pricing | string | "Free personal, $50/yr commercial" |
| status | enum | active, evaluated, abandoned, recommended |
| alternatives | list | ["Logseq", "Notion"] |
| my_rating | int | 1-5 |
| notes | text | "Local-first, markdown, great plugin ecosystem" |
| last_used | date | "2026-08-23" |

**Query patterns:**
- "What tools do I use for X?" → tool WHERE category MATCH X
- "What's my opinion on Obsidian?" → tool WHERE name MATCH
- "What alternatives to X exist?" → tool WHERE alternatives CONTAINS X
- "What tools haven't I used recently?" → tool ORDER BY last_used

---

### Domain 17: Accounts & Credentials

Login information and account details.

**Entity: Account**

| Field | Type | Example |
|---|---|---|
| service | string | "GitHub" |
| username | string | "myusername" |
| email | string | "me@email.com" |
| password_location | string | "1Password → Dev accounts" |
| url | string | "https://github.com" |
| category | string | "development", "social", "finance" |
| notes | text | "2FA enabled via Authy" |
| created | date | "2015-03-20" |

**Query patterns:**
- "What's my GitHub username?" → account WHERE service MATCH
- "What dev accounts do I have?" → account WHERE category == development
- "When did I create this account?" → account.created WHERE service MATCH

---

### Domain 18: Books, Media & Entertainment

What you've read, watched, listened to.

**Entity: Media**

| Field | Type | Example |
|---|---|---|
| title | string | "How to Take Smart Notes" |
| type | enum | book, article, video, podcast, movie, tv_show, music |
| author/creator | string | "Sönke Ahrens" |
| status | enum | want_to_consume, in_progress, completed, abandoned |
| rating | int | 1-5 |
| date_completed | date | "2026-08-20" |
| summary | text | "Zettelkasten method for knowledge workers" |
| key_takeaways | list | ["Atomic notes", "Link densely"] |
| would_recommend | bool | true |
| tags | list | ["productivity", "pkm"] |

**Query patterns:**
- "What books have I read this year?" → media WHERE type == book AND date_completed this_year
- "What should I read next?" → media WHERE status == want_to_consume
- "What's the best podcast I've heard?" → media WHERE type == podcast ORDER BY rating
- "What did I watch last night?" → media WHERE date_completed == yesterday

---

### Domain 19: Locations & Places

Places you've been, want to go, or know about.

**Entity: Place**

| Field | Type | Example |
|---|---|---|
| name | string | "Blue Bottle Coffee" |
| type | enum | restaurant, cafe, office, home, landmark, city, country |
| address | string | "123 Main St" |
| location | string | "San Francisco, CA" |
| notes | text | "Best oat milk latte in the city" |
| rating | int | 1-5 |
| last_visited | date | "2026-08-20" |
| tags | list | ["coffee", "work-friendly", "wifi"] |

**Query patterns:**
- "What's a good coffee shop near X?" → place WHERE type MATCH AND location NEAR
- "Where did I eat last week?" → place WHERE type IN [restaurant, cafe] AND last_visited this_week
- "What places do I want to visit?" → place WHERE status == want_to_visit

---

### Domain 20: Habits & Routines

Things you do regularly.

**Entity: Habit**

| Field | Type | Example |
|---|---|---|
| name | string | "Morning meditation" |
| frequency | string | "daily", "weekly", "weekdays" |
| time | string | "07:00" |
| duration | duration | "10min" |
| category | string | "health", "productivity", "learning" |
| streak | int | 12 |
| last_completed | date | "2026-08-23" |
| goal | string | "Build a daily practice" |
| notes | text | "Use Headspace app" |

**Query patterns:**
- "What habits am I tracking?" → habit WHERE status == active
- "What's my current streak for X?" → habit.streak WHERE name MATCH
- "Did I meditate today?" → habit WHERE name MATCH AND last_completed == today

---

### Domain 21: Wishes & Someday/Maybe

Things you want to do someday.

**Entity: Wish**

| Field | Type | Example |
|---|---|---|
| title | string | "Learn to play guitar" |
| category | string | "hobby", "travel", "career", "personal" |
| priority | string | "someday" |
| notes | text | "Acoustic, not electric" |
| date_added | date | "2026-03-15" |
| related_goals | list | ["creative-expression"] |

**Query patterns:**
- "What are my someday/maybe items?" → wish WHERE priority == someday
- "What wishes relate to travel?" → wish WHERE category == travel
- "What wishes have I had for over a year?" → wish WHERE date_added < 1_year_ago

---

## Part 2: Cross-Domain Relationships

The power of the system is in the connections. Every entity can link to every other entity.

```
Person ←── works_on ──→ Project
Person ←── attended ──→ Meeting
Person ←── has_task ──→ Task
Person ←── authored ──→ Learning
Person ←── is_related_to ──→ Person
Person ←── lives_in ──→ Place
Person ←── prefers ──→ Preference

Project ←── has_task ──→ Task
Project ←── uses_tool ──→ Tool
Project ←── has_decision ──→ Decision
Project ←── has_goal ──→ Goal
Project ←── involves_topic ──→ Topic
Project ←── related_to ──→ Project

Task ←── depends_on ──→ Task
Task ←── belongs_to ──→ Project
Task ←── assigned_to ──→ Person
Task ←── derived_from ──→ Meeting

Idea ←── relates_to ──→ Idea
Idea ←── could_become ──→ Project
Idea ←── relates_to ──→ Topic

Learning ←── about_topic ──→ Topic
Learning ←── contradicts ──→ Learning
Learning ←── supports ──→ Decision

Decision ←── resolved_by ──→ Lesson
Decision ←── affects ──→ Project
Decision ←── involves ──→ Person

Meeting ←── produced ──→ Decision
Meeting ←── produced ──→ Task
Meeting ←── attended_by ──→ Person
Meeting ←── about ──→ Project

Lesson ←── learned_from ──→ Meeting
Lesson ←── learned_from ──→ Task
Lesson ←── applies_to ──→ Project
```

---

## Part 3: The Neural Network Architecture

### Multi-Task Model (5-10M params, ~20MB)

```
Input: tokenized text (max 512 tokens)
                    │
                    ▼
┌──────────────────────────────────────────────┐
│           SHARED BACKBONE                     │
│                                               │
│  Token Embedding (vocab=30K, dim=256)        │
│       +                                       │
│  Positional Encoding (learned, max=512)      │
│       │                                       │
│  Transformer Block × 4                        │
│    Multi-Head Attention (4 heads, 256d)      │
│    Feed-Forward (256 → 1024 → 256)           │
│    LayerNorm + Residual                       │
│       │                                       │
│  Global Attention Pool → Context [256]       │
│  Per-Token Output → Tokens [512, 256]        │
└──────────┬──────────────┬────────────────────┘
           │              │
     Context [256]   Tokens [512, 256]
           │              │
    ┌──────┴──────┐  ┌───┴────────────────────────┐
    │             │  │                              │
    ▼             ▼  ▼                              ▼
```

### Head 1: Content Classifier
```
Context [256]
    │
    ▼
Linear(256 → 128) → ReLU → Dropout(0.2)
    │
    ▼
Linear(128 → NUM_TYPES)

Output: probability distribution over note types
Types: observation, task, idea, learning, decision,
       meeting, reminder, person_info, project_update,
       preference, reflection, question, plan, 
       celebration, frustration, gratitude

Expandable: new types can be added, retrained
```

### Head 2: Entity Tagger (BIO)
```
Tokens [512, 256]
    │
    ▼
Linear(256 → 128) → ReLU
    │
    ▼
Linear(128 → NUM_BIO_TAGS)

Output: per-token BIO tags
Tags: O, B-PERSON, I-PERSON, B-PROJECT, I-PROJECT,
      B-TOOL, I-TOOL, B-DATE, I-DATE, B-TIME, I-TIME,
      B-ORG, I-ORG, B-LOCATION, I-LOCATION,
      B-CONCEPT, I-CONCEPT, B-ACTION, I-ACTION,
      B-DEADLINE, I-DEADLINE, B-PRODUCT, I-PRODUCT,
      B-RECIPE, I-RECIPE, B-MEDICATION, I-MEDICATION,
      B-ACCOUNT, I-ACCOUNT, B-SKILL, I-SKILL
```

### Head 3: Priority Predictor
```
Context [256]
    │
    ▼
Linear(256 → 64) → ReLU → Dropout(0.1)
    │
    ▼
Linear(64 → 1) → Sigmoid

Output: priority score 0.0 (lowest) to 1.0 (critical)
```

### Head 4: Urgency Predictor
```
Context [256]
    │
    ▼
Linear(256 → 64) → ReLU
    │
    ▼
Linear(64 → 4) → Softmax

Output: immediate, soon, later, someday
```

### Head 5: Sentiment Analyzer
```
Context [256]
    │
    ▼
Linear(256 → 64) → ReLU
    │
    ▼
Linear(64 → 5) → Softmax

Output: positive, negative, neutral, mixed, uncertain
```

### Head 6: Action Extractor (Seq2Seq)
```
Context [256]
    │
    ▼
Linear(256 → 128) → TransformerDecoder (2 layers, 4 heads)
    │
    ▼
Linear(128 → vocab_size) → Softmax

Output: "fix VAD threshold" (action phrase)
Decoder autoregressively generates the action text
Max output length: 32 tokens
```

### Head 7: Summary Extractor
```
Tokens [512, 256]
    │
    ▼
Linear(256 → 64) → ReLU
    │
    ▼
Linear(64 → 1) → Sigmoid (per-token score)
    │
    ▼
Select top-K tokens by score → extractive summary

Output: most important sentences from the text
K = adaptive based on text length
```

### Head 8: Temporal Resolver
```
Context [256]
    │
    ▼
Linear(256 → 64) → ReLU
    │
    ├──→ Linear(64 → 1) → Sigmoid (has_date: 0/1)
    │
    ├──→ Linear(64 → 3) → Softmax (relative: past/today/future)
    │
    └──→ Linear(64 → 12) → Softmax (month prediction)

Output: {has_date, relative_time, month, confidence}
Combined with rules to resolve "next Friday" → absolute date
```

### Head 9: Topic Classifier (Multi-label)
```
Context [256]
    │
    ▼
Linear(256 → 128) → ReLU
    │
    ▼
Linear(128 → NUM_TOPICS) → Sigmoid

Output: probability for each topic (multi-label)
Topics: technology, health, finance, career, relationships,
        education, entertainment, food, travel, home,
        fitness, creativity, spirituality, politics,
        science, philosophy, gardening, parenting, ...

Expandable as new topics emerge
```

### Head 10: Relationship Detector
```
Context [256] + Entity_Pair [512] (two entity embeddings concatenated)
    │
    ▼
Linear(512 → 128) → ReLU
    │
    ▼
Linear(128 → NUM_RELATIONS) → Softmax

Output: relationship type between two detected entities
Relations: works_on, mentioned_in, created, uses, depends_on,
           related_to, decided_by, learned_from, attended,
           assigned_to, lives_in, prefers, authored, ...
```

---

## Part 4: The Embedding System

### Dual Embedding Strategy

```
Text Embedding (for similarity search):
  sentence-transformers/all-MiniLM-L6-v2
  → 384-dim vector
  → Used for: "find related notes", semantic search

Entity Embedding (for entity linking):
  Custom projection from NN backbone
  → 128-dim vector per entity mention
  → Used for: "is this the same Sarah as before?"
```

### FAISS Index Structure

```
vault_index/
├── notes.faiss          # Note-level embeddings (384d)
├── notes_ids.json       # Maps FAISS index → note file path
├── entities.faiss       # Entity embeddings (128d)
├── entities_ids.json    # Maps FAISS index → entity name
└── metadata.json        # Index stats, last updated
```

---

## Part 5: The LLM Layer (Gemma 3 1B)

### When the LLM Is Called

```
NN Classification Confidence:
  > 0.90  →  NN handles it alone (fast path)
  0.70-0.90 → NN handles classification, LLM validates linking
  < 0.70  →  LLM handles full understanding (slow path)

LLM is also called for:
  - Any summarization request
  - Any "what should I do" query
  - Any cross-vault reasoning
  - Any generative task
  - Disambiguation when NN is uncertain
```

### What the LLM Does

```python
class GemmaUnderstanding:
    """Deep understanding layer using Gemma 3 1B."""
    
    def analyze(self, text: str, context: dict = None) -> dict:
        """Full analysis of a transcription."""
        prompt = f"""
        Analyze this text and extract structured information.
        
        Text: "{text}"
        
        {f"Context from recent notes: {context}" if context else ""}
        
        Extract:
        1. Primary type (task/idea/observation/learning/decision/meeting/reminder/preference/reflection)
        2. All entities with types
        3. Priority (critical/high/medium/low/someday) with reasoning
        4. Urgency (immediate/soon/later/someday)
        5. Action items (if any)
        6. Summary (one sentence)
        7. Related topics
        8. Sentiment
        9. Key insight (if any)
        10. Suggested links to existing concepts
        
        Respond in JSON.
        """
        return self.model.generate(prompt)
    
    def answer_query(self, query: str, vault_context: list) -> str:
        """Answer a question using vault context."""
        prompt = f"""
        Based on the following notes from my knowledge base:
        
        {format_notes(vault_context)}
        
        Answer this question: {query}
        
        Be specific. Cite which note the information comes from.
        """
        return self.model.generate(prompt)
    
    def find_connections(self, new_note: dict, related_notes: list) -> list:
        """Discover non-obvious connections between notes."""
        prompt = f"""
        New note: {new_note}
        
        Existing notes that might be related:
        {format_notes(related_notes)}
        
        What connections exist? What patterns do you see?
        Which notes should be linked and why?
        """
        return self.model.generate(prompt)
```

### Knowledge Distillation: LLM Teaches NN

```python
class DistillationPipeline:
    """The LLM generates labels. The NN learns from them."""
    
    def process_transcription(self, text: str):
        # 1. LLM does deep analysis
        llm_output = self.gemma.analyze(text)
        
        # 2. NN tries to predict the same thing
        nn_output = self.nn.predict(text)
        
        # 3. Log the comparison
        self.log_comparison(text, llm_output, nn_output)
        
        # 4. Write to vault using LLM's analysis
        self.vault_writer.write(llm_output)
        
        # 5. If NN was wrong, add to training buffer
        if nn_output.type != llm_output.type:
            self.training_buffer.append({
                "text": text,
                "label": llm_output,  # LLM is the teacher
                "predicted": nn_output
            })
        
        # 6. If NN was confident AND right, mark as "NN can handle"
        if nn_output.confidence > 0.9 and self.matches(llm_output, nn_output):
            self.nn_confident_patterns.append(text)
    
    def retrain_loop(self):
        """Periodic retraining."""
        if len(self.training_buffer) >= 50:
            # Retrain classifier head only (fast)
            self.nn.retrain_head("classifier", self.training_buffer)
        
        if len(self.training_buffer) >= 200:
            # Full model retrain (slower, more thorough)
            self.nn.retrain_all(self.training_buffer)
            # Re-embed all notes if embedding head changed
            self.vault_index.rebuild_embeddings()
```

---

## Part 6: The Vault Structure

```
vault/
├── _index.md                    # Master Map of Content
├── _schema.md                   # Documentation of vault structure
│
├── daily/                       # Daily journals
│   ├── 2026-08-23.md
│   ├── 2026-08-24.md
│   └── ...
│
├── people/                      # Person pages
│   ├── sarah-chen.md
│   ├── mike-johnson.md
│   └── ...
│
├── projects/                    # Project pages
│   ├── voiceassist.md
│   ├── second-brain.md
│   └── ...
│
├── tasks/                       # Task pages
│   ├── fix-vad-threshold.md
│   ├── update-readme.md
│   └── ...
│
├── ideas/                       # Idea pages
│   ├── circular-buffer-pre-roll.md
│   └── ...
│
├── learnings/                   # Learning/research pages
│   ├── zettelkasten-method.md
│   ├── okf-spec.md
│   └── ...
│
├── decisions/                   # Decision logs
│   ├── mlx-over-onnx.md
│   └── ...
│
├── meetings/                    # Meeting notes
│   ├── sprint-planning-2026-08-20.md
│   └── ...
│
├── reminders/                   # Active reminders
│   ├── call-dentist.md
│   └── ...
│
├── preferences/                 # Personal preferences
│   ├── food.md
│   ├── tools.md
│   └── ...
│
├── health/                      # Health records
│   ├── appointments.md
│   └── ...
│
├── finance/                     # Financial records
│   ├── subscriptions.md
│   └── ...
│
├── travel/                      # Travel plans
│   └── ...
│
├── goals/                       # Goals and plans
│   ├── learn-system-design.md
│   └── ...
│
├── lessons/                     # Lessons learned
│   ├── test-before-refactor.md
│   └── ...
│
├── tools/                       # Tool pages
│   ├── obsidian.md
│   ├── pytorch.md
│   └── ...
│
├── accounts/                    # Account records
│   └── ...
│
├── media/                       # Books, articles, etc.
│   ├── how-to-take-smart-notes.md
│   └── ...
│
├── places/                      # Location pages
│   └── ...
│
├── habits/                      # Habit tracking
│   └── ...
│
├── wishes/                      # Someday/maybe
│   └── ...
│
├── topics/                      # Map of Content (MOC) hub notes
│   ├── ai-and-ml.md
│   ├── python.md
│   ├── productivity.md
│   └── ...
│
└── _archive/                    # Completed/old items
    └── ...
```

### Note Template (YAML Frontmatter)

```yaml
---
id: note_2026_08_23_001
type: task
created: 2026-08-23T14:30:00Z
updated: 2026-08-23T14:30:00Z
source: audio
transcription_id: tx_abc123

# Classification
priority: high
urgency: soon
sentiment: neutral
confidence: 0.87

# Entities
people:
  - sarah-chen
projects:
  - voiceassist
tasks:
  - fix-vad-threshold
tools:
  - silero-vad
dates:
  - 2026-08-25

# Organization
tags:
  - bug
  - vad
  - audio
topics:
  - machine-learning
  - signal-processing
category: technical

# Links
related_notes:
  - note_2026_08_20_003
  - note_2026_08_15_001
backlinks: []  # Auto-populated by vault

# Embedding
embedding_id: vec_abc123
---

# Fix VAD Threshold Bug

The Silero VAD threshold is too sensitive, causing false positives
during wake word detection. Need to increase from 0.5 to 0.65.

## Action
Update `vad.py` line 42: `threshold=0.65`

## Context
Related to the latency issue we discussed in the sprint planning
meeting on 2026-08-20.
```

---

## Part 7: The Retrieval System

### Query Processing Pipeline

```
User Query: "What projects am I working on that involve Python?"
                    │
                    ▼
┌──────────────────────────────────────────────────┐
│  STEP 1: PARSE QUERY                             │
│                                                   │
│  Parse with rules + NN:                          │
│  - Intent: list/filter                            │
│  - Entity type: project                           │
│  - Filter: technology = "Python"                  │
│  - Temporal: none                                 │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│  STEP 2: RETRIEVE CANDIDATES                     │
│                                                   │
│  Strategy A: Structural query                    │
│    Search vault/projects/ WHERE technologies      │
│    CONTAINS "Python"                              │
│                                                   │
│  Strategy B: Semantic search                      │
│    Embed query → FAISS nearest neighbors          │
│    in notes.faiss                                 │
│                                                   │
│  Strategy C: Entity graph                         │
│    Find entity "Python" → follow links to projects│
│                                                   │
│  Merge results from all three strategies          │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│  STEP 3: RANK & FILTER                           │
│                                                   │
│  - Filter by relevance score                      │
│  - Deduplicate                                    │
│  - Sort by priority/recency                       │
│  - If < 5 results, expand search                  │
│  - If > 20 results, narrow with more filters      │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│  STEP 4: FORMAT RESPONSE                         │
│                                                   │
│  For simple queries: return structured list       │
│  For complex queries: LLM synthesizes answer      │
│  Always cite source notes                         │
└──────────────────────────────────────────────────┘
```

### Retrieval Strategies

| Strategy | When to Use | How |
|---|---|---|
| **Structural** | Known entity type + known field | Parse frontmatter, filter |
| **Semantic** | Fuzzy/conceptual query | Embedding similarity via FAISS |
| **Graph** | Relationship traversal | Follow wikilinks and entity links |
| **Temporal** | "last week", "today", "this month" | Parse relative time, filter dates |
| **Aggregation** | "how many", "what kind", "summarize" | Collect + compute over results |
| **Hybrid** | Most real queries | Combine 2-3 strategies |

---

## Part 8: The Learning Pipeline

### Continuous Learning Flow

```
┌─────────────────────────────────────────────────────────┐
│                    LEARNING PIPELINE                      │
│                                                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │ Phase 1: Bootstrap (Day 1)                       │    │
│  │                                                   │    │
│  │ Generate 10K-20K synthetic examples:              │    │
│  │ - Template-based text generation                  │    │
│  │ - Synonym replacement                             │    │
│  │ - Speech artifact injection                       │    │
│  │ - Entity injection from seed lists                │    │
│  │ - Priority/urgency label assignment               │    │
│  │                                                   │    │
│  │ Train NN on synthetic data                        │    │
│  │ → 75-80% accuracy                                │    │
│  └─────────────────────────────────────────────────┘    │
│                                                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │ Phase 2: Active Learning (Week 1-4)              │    │
│  │                                                   │    │
│  │ User corrects mistakes:                           │    │
│  │ "That's not a task, it's an idea"                 │    │
│  │ "The priority should be high, not medium"         │    │
│  │ "This person is Sarah, not Susan"                 │    │
│  │                                                   │    │
│  │ Corrections → training buffer                     │    │
│  │ Every 20 corrections → retrain head               │    │
│  │ → 85% accuracy                                   │    │
│  └─────────────────────────────────────────────────┘    │
│                                                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │ Phase 3: Distillation (Month 2+)                 │    │
│  │                                                   │    │
│  │ LLM analyzes every transcription                 │    │
│  │ NN tries to match LLM's analysis                 │    │
│  │ Mismatches → training data                        │    │
│  │ NN learns from LLM's reasoning                    │    │
│  │ → 90% accuracy                                   │    │
│  └─────────────────────────────────────────────────┘    │
│                                                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │ Phase 4: Maturation (Month 6+)                   │    │
│  │                                                   │    │
│  │ NN handles 85%+ of cases                         │    │
│  │ LLM only for deep reasoning                      │    │
│  │ User's vocabulary fully learned                   │    │
│  │ Custom entities recognized                        │    │
│  │ → 95% accuracy                                   │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### Correction Interface

```
After transcription, system shows:
┌─────────────────────────────────────────────────┐
│ 📝 Transcription:                                │
│ "Sarah told me her birthday is March 15th"      │
│                                                  │
│ 🧠 Understood:                                   │
│ Type: observation         [✓] [change]           │
│ Person: Sarah             [✓] [change]           │
│ Attribute: birthday       [✓] [change]           │
│ Value: March 15           [✓] [change]           │
│ Priority: low             [✓] [change]           │
│                                                  │
│ 💾 Save to vault? [Yes] [No] [Edit]             │
└─────────────────────────────────────────────────┘

User clicks [change] on priority → selects "medium"
→ System logs: "User corrected priority from low to medium"
→ This becomes training data
```

---

## Part 9: Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
- [ ] Project structure and module layout
- [ ] Vault directory structure
- [ ] Note writer (markdown + YAML frontmatter)
- [ ] Basic retrieval (structural queries on frontmatter)

### Phase 2: Neural Network Core (Week 3-4)
- [ ] Backbone architecture (4-layer transformer)
- [ ] Head 1: Content classifier
- [ ] Head 2: Entity tagger (BIO)
- [ ] Synthetic data generator
- [ ] Training pipeline
- [ ] Basic inference wrapper

### Phase 3: Embedding & Search (Week 5)
- [ ] sentence-transformers integration (all-MiniLM-L6-v2)
- [ ] FAISS index builder
- [ ] Semantic search
- [ ] Hybrid retrieval (structural + semantic)

### Phase 4: Wire to Voiceassist (Week 6)
- [ ] Hook into transcribe.py pipeline
- [ ] Post-transcription understanding
- [ ] Auto vault writing
- [ ] Basic correction UI

### Phase 5: Advanced Heads (Week 7-8)
- [ ] Head 3-8: Priority, urgency, sentiment, action, summary, temporal
- [ ] Head 9: Multi-label topic classifier
- [ ] Head 10: Relationship detector

### Phase 6: LLM Integration (Week 9-10)
- [ ] Gemma 3 1B integration via MLX
- [ ] Deep understanding layer
- [ ] Knowledge distillation pipeline
- [ ] Query answering with vault context

### Phase 7: Cross-Linking & Graph (Week 11-12)
- [ ] Auto-linking engine (entity resolution)
- [ ] Backlink generation
- [ ] "Related notes" discovery
- [ ] Map of Content (MOC) auto-generation

### Phase 8: Continuous Learning (Week 13-14)
- [ ] Correction logging
- [ ] Training buffer management
- [ ] Periodic retraining
- [ ] Model checkpointing

### Phase 9: Query Interface (Week 15-16)
- [ ] Natural language query parsing
- [ ] Multi-strategy retrieval
- [ ] Response formatting
- [ ] Voice query support ("Hey brain, what projects am I working on?")

---

## Part 10: Dependencies (All Local)

```
# Existing (already in project)
torch>=2.0.0
numpy>=1.24.0
huggingface-hub>=0.20.0
onnxruntime>=1.18.0

# New additions
sentence-transformers>=3.0.0    # Embedding model (all-MiniLM-L6-v2)
faiss-cpu>=1.8.0                # Vector search (CPU only, no GPU needed)
spacy>=3.7.0                    # NER (en_core_web_sm)
mlx>=0.15.0                     # Gemma 3 1B inference (Apple Silicon)
gemma>=0.1.0                    # Gemma model loader
scikit-learn>=1.5.0             # SVM fallback, metrics
pyyaml>=6.0                     # YAML frontmatter parsing
```

All fully local. No cloud services. No API keys. No data leaves the machine.

---

## Part 11: Model Sizes & Performance

| Component | Size | RAM | Inference |
|---|---|---|---|
| Custom NN (5-10M params) | ~20MB | ~50MB | <100ms |
| sentence-transformers | ~80MB | ~200MB | ~20ms |
| spaCy en_core_web_sm | ~12MB | ~50MB | <10ms |
| FAISS index (10K notes) | ~5MB | ~20MB | <5ms |
| Gemma 3 1B | ~2GB | ~3GB | ~1-2s |
| **Total** | **~2.1GB** | **~3.3GB** | **<100ms typical** |

On a MacBook with 8GB+ RAM, this runs comfortably. The custom NN handles daily queries in under 100ms. The LLM is only invoked for complex reasoning.

---

## Summary

This system handles **every type of information a human generates**:

| Domain | Entities | Query Examples |
|---|---|---|
| People | Person | "When is Sarah's birthday?" |
| Projects | Project | "Which projects are pending?" |
| Tasks | Task | "What should I focus on today?" |
| Ideas | Idea | "What ideas do I have about X?" |
| Research | Learning | "What did I learn about Zettelkasten?" |
| Decisions | Decision | "Why did I choose MLX?" |
| Meetings | Meeting | "What was discussed last Tuesday?" |
| Journal | DailyLog | "What did I do this week?" |
| Reminders | Reminder | "What do I need to remember?" |
| Preferences | Preference | "What's my coffee order?" |
| Health | HealthRecord | "When is my dentist appointment?" |
| Finance | FinanceRecord | "What subscriptions do I have?" |
| Travel | TravelRecord | "When is my next trip?" |
| Goals | Goal | "How much progress on my goals?" |
| Lessons | Lesson | "What did I learn from that failure?" |
| Tools | Tool | "What tools do I use for design?" |
| Accounts | Account | "What's my GitHub username?" |
| Media | Media | "What books should I read next?" |
| Places | Place | "Where did I eat last week?" |
| Habits | Habit | "Did I meditate today?" |
| Wishes | Wish | "What are my someday/maybe items?" |

21 domains. Hundreds of fields. Thousands of query patterns. All local. All voice-driven. All improving over time.

**This is a second brain.**
