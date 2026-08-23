"""
brain.data — Synthetic data generator for training.

Generates labeled training examples from templates covering all note types.
Templates model spoken transcriptions with natural speech patterns.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from brain.config import SYNTHETIC_SEED, SYNTHETIC_SAMPLES


# =============================================================================
# Seed data: Templates, entities, and vocabulary for each note type
# =============================================================================

PERSON_NAMES = [
    "Sarah", "Mike", "John", "Emily", "David", "Lisa",
    "Alex", "Jordan", "Chris", "Pat", "Sam", "Kim",
    "Rachel", "Tom", "Maria", "Daniel", "Nina", "Leo",
]

TOOLS = [
    "Obsidian", "PyTorch", "ONNX", "MLX", "Python",
    "Docker", "Git", "VS Code", "PostgreSQL", "Redis",
    "NumPy", "pandas", "FastAPI", "React", "TypeScript",
    "Node.js", "Terraform", "AWS", "GCP", "Flutter",
]

PROJECTS = [
    "voiceassist", "second-brain", "marketing Q3", "website redesign",
    "mobile app", "data pipeline", "ML model", "API gateway",
    "design system", "payment integration", "auth service", "dashboard",
    "cli tool", "monitoring stack", "CI pipeline", "docs site",
]

DEADLINES = [
    "Friday", "next Monday", "end of week", "tomorrow",
    "this afternoon", "next Tuesday", "by EOD", "next week",
    "Wednesday", "the 15th", "end of month", "Saturday",
]

TIMEFRAMES = [
    "morning", "afternoon", "evening", "week", "month",
    "next few days", "this weekend", "next sprint",
]

ACTIONS = [
    "fix", "update", "review", "ship", "test", "deploy",
    "write", "rewrite", "debug", "refactor", "check",
    "build", "install", "configure", "optimize", "measure",
    "document", "clean up", "organize", "schedule", "plan",
    "design", "prototype", "benchmark", "profile", "migrate",
    "integrate", "validate", "present", "draft", "submit",
]

THINGS = [
    "VAD threshold", "wake word model", "README", "test suite",
    "deployment script", "API endpoint", "database migration",
    "authentication flow", "CI pipeline", "documentation",
    "user interface", "performance benchmarks", "error handling",
    "caching layer", "rate limiter", "logging system",
    "backup procedure", "monitoring alerts", "search index",
    "onboarding flow", "billing system", "notification service",
]

LOCATIONS = [
    "the office", "home", "Starbucks", "the library",
    "conference room B", "the park", " coworking space",
    "the gym", "the airport", "downtown",
]

ORGANIZATIONS = [
    "Google", "Apple", "Microsoft", "Amazon", "Meta",
    "the university", "the city council", "the startup",
    "our team", "the research lab", "the design agency",
]

FOOD_ITEMS = [
    "pizza", "sushi", "pasta", "salad", "tacos",
    "Thai food", "Indian food", "Chinese food", "burgers",
    "ramen", "pho", "burritos", "steak", "vegetables",
]

HOBBIES = [
    "reading", "hiking", "cooking", "photography", "guitar",
    "painting", "writing", "running", "yoga", "gaming",
    "gardening", "knitting", "chess", "surfing", "dancing",
]

# =============================================================================
# Templates by note type
# =============================================================================

TASK_TEMPLATES = [
    "I need to {action} the {thing}",
    "TODO: {action} {thing} before {deadline}",
    "don't forget to {action} {thing}",
    "{thing} needs to be {action}",
    "I should {action} {thing} when I get a chance",
    "can you remind me to {action} {thing}",
    "the {thing} needs {action} by {deadline}",
    "I want to {action} the {thing} this {timeframe}",
    "need to get {thing} {action} before {deadline}",
    "quick task: {action} {thing}",
    "I have to {action} {thing} before {deadline}",
    "{thing} is broken, I need to {action} it",
    "let's {action} the {thing} this {timeframe}",
    "action item: {action} {thing}",
    "blocker: can't {action} {thing} until {person} finishes",
    "{person} asked me to {action} {thing}",
    "I'll {action} the {thing} after lunch",
    "we need to {action} the {thing} ASAP",
    "the {thing} {action} is overdue",
    "putting a pin in {thing}, will {action} later",
]

IDEA_TEMPLATES = [
    "what if we {action} a {thing} for the {project}",
    "I just had an idea: {action} the {thing} using {tool}",
    "we should {action} a {thing} in the {project}",
    "what about {action} the {thing} differently",
    "I think we could {action} the {thing} by {timeframe}",
    "cool idea: {action} {thing} with {tool}",
    "imagine if we {action} the {thing}",
    "what if {person} {action} the {thing} for {project}",
    "we could build a {thing} that {action}s automatically",
    "new idea for {project}: {action} a {thing}",
    "what if we combined {tool} with {thing}",
    "brainstorm: {action} a {thing} for the team",
    "I wonder if {tool} could {action} the {thing}",
    "proposal: {action} the {thing} using {tool}",
    "idea for next sprint: {action} {thing}",
]

OBSERVATION_TEMPLATES = [
    "I noticed that the {thing} is {observation}",
    "the {thing} seems to be {observation}",
    "today I observed that {thing} {observation}",
    "interesting: the {thing} {observation}",
    "just realized that {thing} is {observation}",
    "the {thing} has been {observation} lately",
    "noticed {person} is working on the {thing}",
    "the {project} {thing} looks {observation}",
    "I see that the {thing} {observation}",
    "the {tool} output shows {thing} is {observation}",
    "monitoring update: {thing} is now {observation}",
    "the {thing} performance is {observation}",
    "I've been thinking about the {thing}",
    "the {project} dashboard shows {thing} is {observation}",
]

LEARNING_TEMPLATES = [
    "I learned that {tool} can {action} {thing}",
    "today I discovered {thing} about {tool}",
    "the key insight is that {thing} {observation}",
    "I now understand why {thing} is {observation}",
    "lesson learned: {action} the {thing} first",
    "turns out {tool} doesn't {action} {thing} properly",
    "today's learning: {thing} {observation}",
    "I figured out that {thing} {observation}",
    "the documentation says {thing} should be {observation}",
    "from the {project}: always {action} the {thing}",
    "TIL: {tool} has a {action} command for {thing}",
    "I didn't know {tool} could {action} {thing}",
]

DECISION_TEMPLATES = [
    "I decided to {action} the {thing} using {tool}",
    "we're going with {tool} for the {thing}",
    "decision: {action} the {thing} instead of {tool}",
    "I've decided not to {action} the {thing}",
    "final decision on {project}: {action} the {thing}",
    "we chose {tool} over {thing} for {project}",
    "I'm going to {action} the {thing} my way",
    "after thinking about it, I'll {action} {thing}",
    "the team decided to {action} the {thing}",
    "we'll go with {tool} for the {thing}",
    "decision made: {action} the {thing} this {timeframe}",
    "I'm committing to {action} the {thing} before {deadline}",
]

MEETING_TEMPLATES = [
    "meeting with {person} about the {thing}",
    "had a call with {person} regarding {project}",
    "standup update: {thing} is {observation}",
    "sprint planning: we need to {action} the {thing}",
    "1:1 with {person} went well, talked about {thing}",
    "meeting notes: {person} wants to {action} the {thing}",
    "retrospective: the {thing} {observation}",
    "sync with {person} about {project} {thing}",
    "team meeting: decided to {action} the {thing}",
    "brainstorm session for {project} with {person}",
    "discussed the {thing} with {person}, they said {observation}",
    "weekly sync: {project} progress is {observation}",
]

REMINDER_TEMPLATES = [
    "remind me to {action} {thing} at {deadline}",
    "don't forget: {action} {thing} before {deadline}",
    "I need a reminder to {action} {thing} {timeframe}",
    "set a reminder for {deadline}: {action} {thing}",
    "remember to {action} {thing} when you get to {location}",
    "reminder: {person} is coming {timeframe}",
    "don't let me forget to {action} {thing}",
    "alert: {thing} needs {action} by {deadline}",
    "upcoming: {action} {thing} {deadline}",
    "ping me {timeframe} to {action} the {thing}",
]

PERSON_INFO_TEMPLATES = [
    "{person} works at {organization}",
    "{person} is a {role} at {organization}",
    "{person} mentioned they like {hobby}",
    "I met {person} at {location}",
    "{person} is interested in {thing}",
    "reminder: {person}'s birthday is {deadline}",
    "{person} said they're working on {thing}",
    "note about {person}: they prefer {thing}",
    "{person} can help with {thing}",
    "contact info for {person} at {organization}",
    "{person} is the {role} for {project}",
    " talked to {person} about {thing} at {location}",
]

PREFERENCE_TEMPLATES = [
    "I prefer {tool} over {thing} for {task}",
    "my favorite {thing} is {preference}",
    "I like to {action} the {thing} in a specific way",
    "I don't like when {thing} is {observation}",
    "I always use {tool} for {thing}",
    "my preferred workflow for {thing} involves {tool}",
    "I hate it when the {thing} does {observation}",
    "I'd rather {action} than {action2}",
    "the best {thing} is definitely {tool}",
    "I enjoy {hobby} in my free time",
    "I prefer {preference} for {thing}",
    "when it comes to {thing}, I always choose {tool}",
]

JOURNAL_TEMPLATES = [
    "today was a {emotion} day, I worked on {thing}",
    "feeling {emotion} about the {project} progress",
    "today I {action} {thing} and it went {observation}",
    "reflections on {project}: it's {observation}",
    "good day today: {action} the {thing} successfully",
    "rough day: the {thing} broke and I had to {action} it",
    "productive afternoon: {action} {thing} and {thing2}",
    "today I spent most of the day {action}ing {thing}",
    "feeling accomplished: finally {action} the {thing}",
    "late night working on {thing}, feeling {emotion}",
    "grateful for {person} helping with {thing}",
    "today I {observation} and it changed my perspective",
]


# =============================================================================
# Entity injection helpers
# =============================================================================

OBSERVATIONS = [
    "running slowly", "not responding", "working well",
    "using too much memory", "almost ready", "behind schedule",
    "ahead of schedule", "laggy", "smooth", "crashing",
    "over budget", "under budget", "on track",
    "stuck", "progressing", "needs attention",
]

EMOTIONS = [
    "happy", "productive", "tired", "frustrated", "excited",
    "calm", "anxious", "motivated", "stressed", "relaxed",
    "accomplished", "overwhelmed", "focused", "distracted",
]

ROLES = [
    "engineer", "manager", "designer", "lead", "director",
    "intern", "consultant", "founder", "VP", "CTO",
    "developer", "analyst", "architect", "researcher",
]

PREFERENCES = [
    "dark mode", "keyboard shortcuts", "terminal",
    "vim", "VS Code", "minimalist design",
    "structured notes", "bullet points", "long-form writing",
]


# =============================================================================
# Speech artifact injection
# =============================================================================

FILLER_PHRASES = [
    "um", "uh", "so basically", "you know",
    "I mean", "kind of", "sort of", "like",
    "actually", "literally", "right", "okay so",
    "so yeah", "anyway", "hmm", "erm",
]

HEDGE_PHRASES = [
    "I think", "I guess", "probably",
    "maybe", "perhaps", "not sure but",
]

CONFIDENCE_MARKERS = [
    "definitely", "absolutely", "for sure",
    "no doubt", "clearly", "obviously",
]


class SyntheticDataGenerator:
    """
    Generate labeled training examples from templates.

    Produces realistic spoken transcriptions with speech artifacts,
    entity injection, and multi-label annotations.
    """

    def __init__(self, seed: int = SYNTHETIC_SEED):
        self.rng = np.random.RandomState(seed)

    def generate(self, num_samples: int = SYNTHETIC_SAMPLES) -> List[Dict]:
        """
        Generate synthetic training data.

        Args:
            num_samples: Total number of samples to generate.

        Returns:
            List of dicts with text, labels, and metadata.
        """
        samples_per_type = num_samples // 10  # roughly 10 types
        data = []

        # Generate from each template type
        generators = [
            ("task", self._generate_tasks),
            ("idea", self._generate_ideas),
            ("observation", self._generate_observations),
            ("learning", self._generate_learnings),
            ("decision", self._generate_decisions),
            ("meeting", self._generate_meetings),
            ("reminder", self._generate_reminders),
            ("person_info", self._generate_person_info),
            ("preference", self._generate_preferences),
            ("journal", self._generate_journals),
        ]

        for note_type, gen_fn in generators:
            type_data = gen_fn(samples_per_type)
            data.extend(type_data)

        # Shuffle
        self.rng.shuffle(data)

        return data

    def _fill_template(
        self,
        template: str,
        replacements: Dict[str, str],
    ) -> str:
        """Fill entity placeholders in a template."""
        result = template
        for key, value in replacements.items():
            result = result.replace("{" + key + "}", value)
        return result

    def _inject_speech_artifacts(self, text: str) -> str:
        """Add speech artifacts to make text sound more natural."""
        if self.rng.random() < 0.3:
            # Prepend filler
            filler = self.rng.choice(FILLER_PHRASES)
            text = f"{filler}, {text}"

        if self.rng.random() < 0.2:
            # Add hedge phrase
            hedge = self.rng.choice(HEDGE_PHRASES)
            text = f"{hedge}, {text}"

        return text

    def _add_noise(self, text: str) -> str:
        """Add minor noise: capitalization, punctuation variations."""
        # Sometimes make first letter lowercase (like speech)
        if self.rng.random() < 0.4:
            text = text[0].lower() + text[1:]

        # Sometimes remove trailing period
        if self.rng.random() < 0.3:
            text = text.rstrip(".")

        return text

    def _make_sample(
        self,
        text: str,
        note_type: str,
        priority: float,
        urgency: str,
        sentiment: str,
        entities: Optional[List[Dict]] = None,
        action: Optional[str] = None,
        topics: Optional[List[str]] = None,
        has_temporal: bool = False,
    ) -> Dict:
        """Construct a training sample dict."""
        text = self._inject_speech_artifacts(text)
        text = self._add_noise(text)

        if entities is None:
            entities = []
        if topics is None:
            topics = []

        return {
            "text": text,
            "type": note_type,
            "priority": priority,
            "urgency": urgency,
            "sentiment": sentiment,
            "entities": entities,
            "action": action,
            "topics": topics,
            "has_temporal": has_temporal,
        }

    def _random_person(self) -> str:
        return self.rng.choice(PERSON_NAMES)

    def _random_tool(self) -> str:
        return self.rng.choice(TOOLS)

    def _random_project(self) -> str:
        return self.rng.choice(PROJECTS)

    def _random_deadline(self) -> str:
        return self.rng.choice(DEADLINES)

    def _random_timeframe(self) -> str:
        return self.rng.choice(TIMEFRAMES)

    def _random_action(self) -> str:
        return self.rng.choice(ACTIONS)

    def _random_thing(self) -> str:
        return self.rng.choice(THINGS)

    def _random_location(self) -> str:
        return self.rng.choice(LOCATIONS)

    def _random_organization(self) -> str:
        return self.rng.choice(ORGANIZATIONS)

    def _random_hobby(self) -> str:
        return self.rng.choice(HOBBIES)

    def _random_observation(self) -> str:
        return self.rng.choice(OBSERVATIONS)

    def _random_emotion(self) -> str:
        return self.rng.choice(EMOTIONS)

    def _random_role(self) -> str:
        return self.rng.choice(ROLES)

    def _random_preference(self) -> str:
        return self.rng.choice(PREFERENCES)

    def _random_topics(self, n: int = 2) -> List[str]:
        from brain_contract import TOPICS
        return list(self.rng.choice(TOPICS, size=n, replace=False))

    # -------------------------------------------------------------------------
    # Per-type generators
    # -------------------------------------------------------------------------

    def _generate_tasks(self, n: int) -> List[Dict]:
        data = []
        for _ in range(n):
            person = self._random_person()
            tool = self._random_tool()
            action = self._random_action()
            thing = self._random_thing()
            deadline = self._random_deadline()
            timeframe = self._random_timeframe()

            template = self.rng.choice(TASK_TEMPLATES)
            text = self._fill_template(template, {
                "person": person, "tool": tool, "action": action,
                "thing": thing, "deadline": deadline, "timeframe": timeframe,
                "project": self._random_project(),
                "location": self._random_location(),
            })

            entities = []
            if self.rng.random() < 0.6:
                entities.append({"name": thing, "type": "tool"})
            if self.rng.random() < 0.7:
                entities.append({"name": deadline, "type": "date"})
            if self.rng.random() < 0.3:
                entities.append({"name": person, "type": "person"})
            if self.rng.random() < 0.4:
                entities.append({"name": self._random_project(), "type": "project"})

            priority = float(self.rng.uniform(0.5, 1.0))
            urgency = self.rng.choice(
                ["soon", "immediate", "later"],
                p=[0.5, 0.3, 0.2],
            )

            data.append(self._make_sample(
                text=text,
                note_type="task",
                priority=priority,
                urgency=urgency,
                sentiment="neutral",
                entities=entities,
                action=action,
                topics=self._random_topics(2),
                has_temporal=True,
            ))
        return data

    def _generate_ideas(self, n: int) -> List[Dict]:
        data = []
        for _ in range(n):
            person = self._random_person()
            tool = self._random_tool()
            action = self._random_action()
            thing = self._random_thing()
            project = self._random_project()
            timeframe = self._random_timeframe()

            template = self.rng.choice(IDEA_TEMPLATES)
            text = self._fill_template(template, {
                "person": person, "tool": tool, "action": action,
                "thing": thing, "project": project, "timeframe": timeframe,
            })

            entities = []
            if self.rng.random() < 0.5:
                entities.append({"name": project, "type": "project"})
            if self.rng.random() < 0.6:
                entities.append({"name": tool, "type": "tool"})
            if self.rng.random() < 0.3:
                entities.append({"name": person, "type": "person"})

            data.append(self._make_sample(
                text=text,
                note_type="idea",
                priority=float(self.rng.uniform(0.3, 0.7)),
                urgency="later",
                sentiment="positive",
                entities=entities,
                action=action,
                topics=self._random_topics(2),
            ))
        return data

    def _generate_observations(self, n: int) -> List[Dict]:
        data = []
        for _ in range(n):
            thing = self._random_thing()
            observation = self._random_observation()
            person = self._random_person()
            tool = self._random_tool()
            project = self._random_project()

            template = self.rng.choice(OBSERVATION_TEMPLATES)
            text = self._fill_template(template, {
                "thing": thing, "observation": observation,
                "person": person, "tool": tool, "project": project,
            })

            entities = []
            if self.rng.random() < 0.4:
                entities.append({"name": thing, "type": "tool"})
            if self.rng.random() < 0.3:
                entities.append({"name": person, "type": "person"})

            sentiment = self.rng.choice(
                ["positive", "negative", "neutral"],
                p=[0.3, 0.3, 0.4],
            )

            data.append(self._make_sample(
                text=text,
                note_type="observation",
                priority=float(self.rng.uniform(0.1, 0.5)),
                urgency="later",
                sentiment=sentiment,
                entities=entities,
                topics=self._random_topics(1),
            ))
        return data

    def _generate_learnings(self, n: int) -> List[Dict]:
        data = []
        for _ in range(n):
            tool = self._random_tool()
            action = self._random_action()
            thing = self._random_thing()
            project = self._random_project()
            observation = self._random_observation()

            template = self.rng.choice(LEARNING_TEMPLATES)
            text = self._fill_template(template, {
                "tool": tool, "action": action, "thing": thing,
                "project": project, "observation": observation,
            })

            entities = []
            if self.rng.random() < 0.6:
                entities.append({"name": tool, "type": "tool"})
            if self.rng.random() < 0.4:
                entities.append({"name": project, "type": "project"})

            data.append(self._make_sample(
                text=text,
                note_type="learning",
                priority=float(self.rng.uniform(0.2, 0.6)),
                urgency="later",
                sentiment="neutral",
                entities=entities,
                action=action,
                topics=self._random_topics(2),
            ))
        return data

    def _generate_decisions(self, n: int) -> List[Dict]:
        data = []
        for _ in range(n):
            tool = self._random_tool()
            action = self._random_action()
            thing = self._random_thing()
            project = self._random_project()
            timeframe = self._random_timeframe()
            deadline = self._random_deadline()

            template = self.rng.choice(DECISION_TEMPLATES)
            text = self._fill_template(template, {
                "tool": tool, "action": action, "thing": thing,
                "project": project, "timeframe": timeframe, "deadline": deadline,
            })

            entities = []
            if self.rng.random() < 0.5:
                entities.append({"name": tool, "type": "tool"})
            if self.rng.random() < 0.5:
                entities.append({"name": project, "type": "project"})

            data.append(self._make_sample(
                text=text,
                note_type="decision",
                priority=float(self.rng.uniform(0.5, 0.9)),
                urgency="soon",
                sentiment="neutral",
                entities=entities,
                action=action,
                topics=self._random_topics(2),
            ))
        return data

    def _generate_meetings(self, n: int) -> List[Dict]:
        data = []
        for _ in range(n):
            person = self._random_person()
            thing = self._random_thing()
            project = self._random_project()
            action = self._random_action()
            observation = self._random_observation()

            template = self.rng.choice(MEETING_TEMPLATES)
            text = self._fill_template(template, {
                "person": person, "thing": thing, "project": project,
                "action": action, "observation": observation,
            })

            entities = []
            if self.rng.random() < 0.8:
                entities.append({"name": person, "type": "person"})
            if self.rng.random() < 0.5:
                entities.append({"name": project, "type": "project"})

            data.append(self._make_sample(
                text=text,
                note_type="meeting",
                priority=float(self.rng.uniform(0.4, 0.8)),
                urgency="soon",
                sentiment="neutral",
                entities=entities,
                action=action,
                topics=self._random_topics(1),
                has_temporal=True,
            ))
        return data

    def _generate_reminders(self, n: int) -> List[Dict]:
        data = []
        for _ in range(n):
            person = self._random_person()
            action = self._random_action()
            thing = self._random_thing()
            deadline = self._random_deadline()
            timeframe = self._random_timeframe()
            location = self._random_location()

            template = self.rng.choice(REMINDER_TEMPLATES)
            text = self._fill_template(template, {
                "person": person, "action": action, "thing": thing,
                "deadline": deadline, "timeframe": timeframe,
                "location": location,
            })

            entities = []
            if self.rng.random() < 0.7:
                entities.append({"name": deadline, "type": "date"})
            if self.rng.random() < 0.3:
                entities.append({"name": person, "type": "person"})

            data.append(self._make_sample(
                text=text,
                note_type="reminder",
                priority=float(self.rng.uniform(0.6, 1.0)),
                urgency="immediate",
                sentiment="neutral",
                entities=entities,
                action=action,
                topics=self._random_topics(1),
                has_temporal=True,
            ))
        return data

    def _generate_person_info(self, n: int) -> List[Dict]:
        data = []
        for _ in range(n):
            person = self._random_person()
            organization = self._random_organization()
            role = self._random_role()
            thing = self._random_thing()
            hobby = self._random_hobby()
            project = self._random_project()
            location = self._random_location()
            deadline = self._random_deadline()

            template = self.rng.choice(PERSON_INFO_TEMPLATES)
            text = self._fill_template(template, {
                "person": person, "organization": organization,
                "role": role, "thing": thing, "hobby": hobby,
                "project": project, "location": location,
                "deadline": deadline,
            })

            entities = [{"name": person, "type": "person"}]
            if self.rng.random() < 0.5:
                entities.append({"name": organization, "type": "organization"})
            if self.rng.random() < 0.3:
                entities.append({"name": project, "type": "project"})

            data.append(self._make_sample(
                text=text,
                note_type="person_info",
                priority=float(self.rng.uniform(0.2, 0.6)),
                urgency="someday",
                sentiment="neutral",
                entities=entities,
                topics=["relationships"],
            ))
        return data

    def _generate_preferences(self, n: int) -> List[Dict]:
        data = []
        for _ in range(n):
            tool = self._random_tool()
            thing = self._random_thing()
            action = self._random_action()
            hobby = self._random_hobby()
            preference = self._random_preference()
            observation = self._random_observation()
            task = self.rng.choice(["coding", "writing", "design", "planning"])

            template = self.rng.choice(PREFERENCE_TEMPLATES)
            text = self._fill_template(template, {
                "tool": tool, "thing": thing, "action": action,
                "hobby": hobby, "preference": preference,
                "observation": observation, "task": task,
                "action2": self._random_action(),
            })

            entities = []
            if self.rng.random() < 0.5:
                entities.append({"name": tool, "type": "tool"})

            data.append(self._make_sample(
                text=text,
                note_type="preference",
                priority=float(self.rng.uniform(0.1, 0.4)),
                urgency="someday",
                sentiment=self.rng.choice(["positive", "negative"]),
                entities=entities,
                topics=["productivity"],
            ))
        return data

    def _generate_journals(self, n: int) -> List[Dict]:
        data = []
        for _ in range(n):
            emotion = self._random_emotion()
            thing = self._random_thing()
            thing2 = self._random_thing()
            project = self._random_project()
            action = self._random_action()
            observation = self._random_observation()
            person = self._random_person()

            template = self.rng.choice(JOURNAL_TEMPLATES)
            text = self._fill_template(template, {
                "emotion": emotion, "thing": thing, "thing2": thing2,
                "project": project, "action": action,
                "observation": observation, "person": person,
            })

            entities = []
            if self.rng.random() < 0.3:
                entities.append({"name": project, "type": "project"})
            if self.rng.random() < 0.2:
                entities.append({"name": person, "type": "person"})

            sentiment = "positive" if emotion in (
                "happy", "productive", "excited", "motivated",
                "accomplished", "focused",
            ) else "negative" if emotion in (
                "frustrated", "anxious", "stressed",
                "overwhelmed", "tired",
            ) else "neutral"

            data.append(self._make_sample(
                text=text,
                note_type="journal",
                priority=float(self.rng.uniform(0.1, 0.4)),
                urgency="someday",
                sentiment=sentiment,
                entities=entities,
                topics=self._random_topics(1),
            ))
        return data

    # -------------------------------------------------------------------------
    # Save / Load
    # -------------------------------------------------------------------------

    def save(self, data: List[Dict], path: str) -> None:
        """Save generated data to JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load(path: str) -> List[Dict]:
        """Load generated data from JSON file."""
        with open(path, "r") as f:
            return json.load(f)


# =============================================================================
# Convenience function
# =============================================================================

def generate_training_data(
    num_samples: int = SYNTHETIC_SAMPLES,
    seed: int = SYNTHETIC_SEED,
) -> List[Dict]:
    """Quick generate + return."""
    gen = SyntheticDataGenerator(seed=seed)
    return gen.generate(num_samples)
