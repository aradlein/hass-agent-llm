"""Unit tests for whole-house question detection and context expansion."""

import json
from unittest.mock import Mock, patch

import pytest
from homeassistant.core import State

from custom_components.home_agent.const import CONF_OPENAI_API_KEY
from custom_components.home_agent.context_providers.vector_db import (
    CHROMADB_AVAILABLE,
    WHOLE_HOUSE_MAX_ENTITIES,
    VectorDBContextProvider,
)
from custom_components.home_agent.context_providers.whole_house import (
    NO_FLOOR,
    detect_whole_house_query,
    normalize_state,
    summarize_devices,
)

EXPOSE = "custom_components.home_agent.context_providers.vector_db.async_should_expose"


def _domains(query: str) -> set[str]:
    return {s.domain for s in detect_whole_house_query(query)[1]}


class TestDetectWholeHouseQuery:
    """Tests for detect_whole_house_query."""

    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("which lights are on?", {"light"}),
            ("Which lights are on", {"light"}),
            ("are any lights on", {"light"}),
            ("what lights are on right now", {"light"}),
            ("did I leave any lights on?", {"light"}),
            ("how many lights are on", {"light"}),
            ("are all the lights off", {"light"}),
            ("is any light on upstairs", {"light"}),
            ("can you tell me which lamps are on", {"light"}),
            ("are any doors unlocked?", {"lock", "cover", "binary_sensor"}),
            ("are the doors locked", {"lock", "cover", "binary_sensor"}),
            ("is every door locked", {"lock", "cover", "binary_sensor"}),
            ("which blinds are open", {"cover"}),
            ("are any windows open", {"binary_sensor", "cover"}),
            ("what are the thermostats set to", {"climate"}),
            ("are any fans running", {"fan"}),
            ("is anything on?", {"light", "switch", "fan"}),
            ("is everything locked", {"lock", "cover", "binary_sensor"}),
            ("is anything open", {"cover", "binary_sensor"}),
            ("are any lights or fans on", {"light", "fan"}),
        ],
    )
    def test_whole_house_questions(self, query, expected):
        """Whole-category questions select the matching domains."""
        assert _domains(query) == expected

    @pytest.mark.parametrize(
        "query",
        [
            "is the office light on?",
            "is the front door locked",
            "what is the thermostat set to upstairs",
            "what's the temperature in the living room",
            "is the garage door closed",
            "tell me a joke",
            "what time is it",
            "what's on the calendar today",
            "",
        ],
    )
    def test_single_device_and_other_questions(self, query):
        """Single-device and unrelated questions are left alone."""
        assert detect_whole_house_query(query) == ([], [])

    @pytest.mark.parametrize(
        "query",
        [
            "turn off all the lights",
            "please turn on all the lights",
            "can you turn off all the lights",
            "hey jarvis, turn off all the lights",
            "lock all the doors",
            "close all the blinds",
            "set all the thermostats to 70",
            "I want you to turn off all the lights",
            "go ahead and lock all the doors",
            "I'd like you to close all the blinds",
        ],
    )
    def test_commands_are_not_expanded(self, query):
        """Commands keep today's context."""
        assert detect_whole_house_query(query) == ([], [])

    def test_labels_name_the_categories(self):
        """Labels describe what was matched."""
        labels, _ = detect_whole_house_query("which lights are on")
        assert labels == ["lights"]

    def test_device_class_narrowing(self):
        """Door questions take door sensors but not motion sensors."""
        _, selectors = detect_whole_house_query("are any doors open")
        assert any(s.matches("binary_sensor", "door") for s in selectors)
        assert not any(s.matches("binary_sensor", "motion") for s in selectors)
        assert any(s.matches("lock", None) for s in selectors)


class TestOtherLanguages:
    """The detector works in the other languages it has a vocabulary for."""

    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            # German
            ("Welche Lichter sind an?", {"light"}),
            ("Wie viele Lampen sind an", {"light"}),
            ("Sind alle Türen abgeschlossen?", {"lock", "cover", "binary_sensor"}),
            ("Sind die Fenster zu?", {"binary_sensor", "cover"}),
            ("Ist irgendetwas offen?", {"cover", "binary_sensor"}),
            ("Welche Rollläden sind offen", {"cover"}),
            ("Sind die Fernseher aus?", {"media_player"}),
            # French
            ("Quelles lumières sont allumées ?", {"light"}),
            ("Est-ce qu'il y a des lumières allumées ?", {"light"}),
            ("Combien de lampes sont allumées", {"light"}),
            ("Toutes les portes sont-elles verrouillées ?", {"lock", "cover", "binary_sensor"}),
            ("Est-ce que quelque chose est ouvert ?", {"cover", "binary_sensor"}),
            ("Quels volets sont ouverts", {"cover"}),
            # Spanish
            ("¿Qué luces están encendidas?", {"light"}),
            ("¿Cuántas luces están encendidas?", {"light"}),
            ("¿Hay alguna ventana abierta?", {"binary_sensor", "cover"}),
            ("¿Hay luces encendidas?", {"light"}),
            ("¿Están todas las puertas bloqueadas?", {"lock", "cover", "binary_sensor"}),
            ("¿Hay algo encendido?", {"light", "switch", "fan"}),
            # Dutch
            ("Welke lampen staan aan?", {"light"}),
            ("Staan er nog lampen aan", {"light"}),
            ("Zijn alle deuren op slot?", {"lock", "cover", "binary_sensor"}),
            ("Staat er iets aan?", {"light", "switch", "fan"}),
            ("Welke ramen staan open", {"binary_sensor", "cover"}),
            # Italian
            ("Quali luci sono accese?", {"light"}),
            ("C'è qualche luce accesa?", {"light"}),
            ("Ci sono luci accese?", {"light"}),
            ("Tutte le porte sono bloccate?", {"lock", "cover", "binary_sensor"}),
            ("C'è qualcosa di aperto?", {"cover", "binary_sensor"}),
            ("Quante finestre sono aperte", {"binary_sensor", "cover"}),
        ],
    )
    def test_whole_house_questions(self, query, expected):
        """Whole-category questions select the matching domains."""
        assert _domains(query) == expected

    @pytest.mark.parametrize(
        "query",
        [
            "Ist das Licht im Büro an?",
            "Ist die Haustür abgeschlossen",
            "La lumière du bureau est-elle allumée ?",
            "Est-ce que la porte d'entrée est verrouillée",
            "¿Está encendida la luz de la oficina?",
            "¿Está cerrada la puerta principal?",
            "Is het licht in het kantoor aan?",
            "Is de voordeur op slot",
            "La luce dell'ufficio è accesa?",
            "La porta d'ingresso è chiusa",
            # Same spelling for one and many: only the plural article makes it plural
            "Was läuft auf dem Fernseher?",
            "Was ist mit dem Fenster im Bad",
            # "Is there ..." about one thing
            "Il y a quelqu'un à la porte ?",
            "¿Hay alguien en la puerta?",
            "C'è qualcuno alla porta?",
        ],
    )
    def test_single_device_questions(self, query):
        """Single-device questions are left alone in every language."""
        assert detect_whole_house_query(query) == ([], [])

    @pytest.mark.parametrize(
        "query",
        [
            "Schalte alle Lichter aus",
            "Bitte schalte alle Lichter aus",
            "Kannst du bitte alle Lichter ausschalten",
            "Schließe alle Rollläden",
            "Éteins toutes les lumières",
            "Peux-tu éteindre toutes les lumières",
            "Ferme tous les volets",
            "Apaga todas las luces",
            "Por favor cierra todas las persianas",
            "Doe alle lampen uit",
            "Kun je alle lampen uitdoen",
            "Spegni tutte le luci",
            "Per favore chiudi tutte le tapparelle",
        ],
    )
    def test_commands_are_not_expanded(self, query):
        """Commands keep today's context in every language."""
        assert detect_whole_house_query(query) == ([], [])

    def test_labels_are_the_same_in_every_language(self):
        """Labels are for the prompt, so they do not depend on the request language."""
        assert detect_whole_house_query("Welche Lichter sind an?")[0] == ["lights"]
        assert detect_whole_house_query("¿Qué luces están encendidas?")[0] == ["lights"]

    def test_accents_are_optional(self):
        """Typed requests often drop accents; speech-to-text keeps them."""
        assert _domains("quelles lumieres sont allumees") == {"light"}
        assert _domains("sind alle turen abgeschlossen") == {"lock", "cover", "binary_sensor"}

    def test_words_from_two_languages_do_not_combine(self):
        """An English quantifier with a Dutch noun is not a match."""
        assert detect_whole_house_query("which slot is free") == ([], [])

    @pytest.mark.parametrize("query", ["どの照明がついていますか", "Które światła są włączone?"])
    def test_language_without_vocabulary_is_left_alone(self, query):
        """A language with no vocabulary gets the normal context, not an error."""
        assert detect_whole_house_query(query) == ([], [])


class TestNormalizeState:
    """Tests for normalize_state."""

    @pytest.mark.parametrize(
        ("domain", "device_class", "state", "expected"),
        [
            ("binary_sensor", "door", "on", "open"),
            ("binary_sensor", "door", "off", "closed"),
            ("binary_sensor", "window", "on", "open"),
            ("binary_sensor", "garage_door", "off", "closed"),
            ("binary_sensor", "opening", "on", "open"),
            ("binary_sensor", "lock", "on", "unlocked"),
            ("binary_sensor", "lock", "off", "locked"),
            ("binary_sensor", "door", "unavailable", "unavailable"),
            ("binary_sensor", "motion", "on", "on"),
            ("binary_sensor", None, "on", "on"),
            ("light", None, "on", "on"),
            ("cover", "door", "open", "open"),
            ("lock", None, "locked", "locked"),
        ],
    )
    def test_states(self, domain, device_class, state, expected):
        """Door, window and lock sensors use the words covers and locks use."""
        assert normalize_state(domain, device_class, state) == expected


def _state(entity_id: str, state: str, **attributes) -> Mock:
    mock = Mock(spec=State)
    mock.entity_id = entity_id
    mock.state = state
    mock.attributes = attributes
    return mock


@pytest.fixture
def provider(mock_hass):
    """Provider with ChromaDB and embeddings stubbed out."""
    if not CHROMADB_AVAILABLE:
        pytest.skip("ChromaDB not available")
    provider = VectorDBContextProvider(mock_hass, {CONF_OPENAI_API_KEY: "sk-test"})
    provider._client = Mock()
    provider._collection = Mock()
    return provider


def _install_states(mock_hass, states):
    by_id = {s.entity_id: s for s in states}
    mock_hass.states.async_all.return_value = states
    mock_hass.states.get.side_effect = by_id.get


HOUSE = [
    _state("light.office", "off", friendly_name="Office Light", brightness=10),
    _state("light.basement", "on", friendly_name="Basement Light"),
    _state("light.hidden", "on", friendly_name="Hidden Light"),
    _state("lock.front_door", "unlocked", friendly_name="Front Door"),
    _state("binary_sensor.motion", "on", friendly_name="Motion", device_class="motion"),
    _state(
        "climate.upstairs",
        "heat",
        friendly_name="Upstairs",
        temperature=69,
        current_temperature=68,
        hvac_modes=["off", "heat"],
    ),
]


class TestWholeHouseContext:
    """Tests for whole-house expansion inside get_context."""

    @pytest.mark.asyncio
    async def test_adds_every_exposed_entity_of_the_domain(self, provider, mock_hass):
        """A lights question gets all exposed lights, beyond the top-k."""
        _install_states(mock_hass, HOUSE)
        with (
            patch.object(provider, "_embed_query", return_value=[0.1]),
            patch.object(
                provider,
                "_query_vector_db",
                return_value=[{"entity_id": "light.office", "distance": 0.1}],
            ),
            patch(EXPOSE, side_effect=lambda hass, assistant, eid: eid != "light.hidden"),
        ):
            parsed = json.loads(await provider.get_context("which lights are on?"))

        assert [e["entity_id"] for e in parsed["entities"]] == ["light.office"]
        extra = parsed["all_matching_devices"]
        assert "lights" in extra["description"]
        assert extra["total"] == 2
        assert extra["by_state"] == {
            "on": {"count": 1, "by_floor": {NO_FLOOR: ["Basement Light"]}},
            "off": {"count": 1, "by_floor": {NO_FLOOR: ["Office Light"]}},
        }

    @pytest.mark.asyncio
    async def test_compact_records_keep_domain_attributes_only(self, provider, mock_hass):
        """Thermostat records carry the target and current temperature, nothing bulky."""
        _install_states(mock_hass, HOUSE)
        with (
            patch.object(provider, "_embed_query", return_value=[0.1]),
            patch.object(provider, "_query_vector_db", return_value=[]),
            patch(EXPOSE, return_value=True),
        ):
            parsed = json.loads(await provider.get_context("what are the thermostats set to"))

        assert parsed["entities"] == []
        assert parsed["all_matching_devices"]["by_state"] == {
            "heat": {
                "count": 1,
                "by_floor": {NO_FLOOR: ["Upstairs, temperature 69, current temperature 68"]},
            }
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize("query", ["is the office light on?", "turn off all the lights"])
    async def test_other_requests_unchanged(self, provider, mock_hass, query):
        """Single-device questions and commands produce today's context."""
        _install_states(mock_hass, HOUSE)
        with (
            patch.object(provider, "_embed_query", return_value=[0.1]),
            patch.object(
                provider,
                "_query_vector_db",
                return_value=[{"entity_id": "light.office", "distance": 0.1}],
            ),
            patch(EXPOSE, return_value=True),
        ):
            parsed = json.loads(await provider.get_context(query))

        assert set(parsed) == {"entities", "count"}
        assert parsed["count"] == 1

    @pytest.mark.asyncio
    async def test_expansion_failure_keeps_normal_context(self, provider, mock_hass):
        """An error while expanding must not lose the semantic-search context."""
        _install_states(mock_hass, HOUSE)
        mock_hass.states.async_all.side_effect = RuntimeError("boom")
        with (
            patch.object(provider, "_embed_query", return_value=[0.1]),
            patch.object(
                provider,
                "_query_vector_db",
                return_value=[{"entity_id": "light.office", "distance": 0.1}],
            ),
        ):
            parsed = json.loads(await provider.get_context("which lights are on?"))

        assert set(parsed) == {"entities", "count"}

    @pytest.mark.asyncio
    async def test_large_home_counts_are_exact_and_names_are_limited(self, provider, mock_hass):
        """Past the cap the counts stay exact and the model is told the list is partial."""
        off = WHOLE_HOUSE_MAX_ENTITIES + 5
        many = [_state(f"light.l{i:03d}", "off", friendly_name=f"L{i}") for i in range(off)]
        many += [_state(f"light.on{i}", "on", friendly_name=f"On{i}") for i in range(3)]
        _install_states(mock_hass, many)
        with (
            patch.object(provider, "_embed_query", return_value=[0.1]),
            patch.object(provider, "_query_vector_db", return_value=[]),
            patch(EXPOSE, return_value=True),
        ):
            parsed = json.loads(await provider.get_context("which lights are on?"))

        extra = parsed["all_matching_devices"]
        assert extra["total"] == off + 3
        assert extra["truncated"] is True
        assert "partial" in extra["description"]
        assert "Every one of" not in extra["description"]
        # The small group is complete; the large one says how many names are missing
        on_group = extra["by_state"]["on"]
        assert on_group["count"] == 3
        assert "not_listed" not in on_group
        assert on_group["by_floor"] == {NO_FLOOR: ["On0", "On1", "On2"]}
        off_group = extra["by_state"]["off"]
        assert off_group["count"] == off
        assert off_group["not_listed"] == 8
        assert len(off_group["by_floor"][NO_FLOOR]) == WHOLE_HOUSE_MAX_ENTITIES - 3

    @pytest.mark.asyncio
    async def test_home_within_cap_is_not_marked_truncated(self, provider, mock_hass):
        """A complete list keeps the wording that it is complete."""
        _install_states(mock_hass, HOUSE)
        with (
            patch.object(provider, "_embed_query", return_value=[0.1]),
            patch.object(provider, "_query_vector_db", return_value=[]),
            patch(EXPOSE, return_value=True),
        ):
            parsed = json.loads(await provider.get_context("which lights are on?"))

        extra = parsed["all_matching_devices"]
        assert "truncated" not in extra
        assert extra["description"].startswith("Every one of the lights")

    @pytest.mark.asyncio
    async def test_door_sensors_and_covers_share_one_vocabulary(self, provider, mock_hass):
        """A sensor that is 'on' is grouped with the cover that is 'open'."""
        _install_states(
            mock_hass,
            [
                _state(
                    "binary_sensor.back_door", "on", friendly_name="Back Door", device_class="door"
                ),
                _state(
                    "binary_sensor.side_door", "off", friendly_name="Side Door", device_class="door"
                ),
                _state("binary_sensor.motion", "on", friendly_name="Motion", device_class="motion"),
                _state("cover.garage", "open", friendly_name="Garage", device_class="garage"),
                _state("cover.blind", "closed", friendly_name="Blind", device_class="blind"),
            ],
        )
        with (
            patch.object(provider, "_embed_query", return_value=[0.1]),
            patch.object(provider, "_query_vector_db", return_value=[]),
            patch(EXPOSE, return_value=True),
        ):
            parsed = json.loads(await provider.get_context("is anything open?"))

        assert parsed["all_matching_devices"]["by_state"] == {
            "open": {"count": 2, "by_floor": {NO_FLOOR: ["Back Door", "Garage"]}},
            "closed": {"count": 2, "by_floor": {NO_FLOOR: ["Side Door", "Blind"]}},
        }

    @pytest.mark.asyncio
    async def test_question_in_another_language(self, provider, mock_hass):
        """A German question gets the same expansion as the English one."""
        _install_states(mock_hass, HOUSE)
        with (
            patch.object(provider, "_embed_query", return_value=[0.1]),
            patch.object(provider, "_query_vector_db", return_value=[]),
            patch(EXPOSE, return_value=True),
        ):
            parsed = json.loads(await provider.get_context("Welche Lichter sind an?"))

        assert parsed["all_matching_devices"]["total"] == 3

    @pytest.mark.asyncio
    async def test_compact_records_include_area_and_floor(self, provider, mock_hass):
        """Area and floor are carried into the compact records."""
        _install_states(mock_hass, HOUSE)
        full = {
            "entity_id": "light.basement",
            "state": "on",
            "attributes": {"friendly_name": "Basement Light"},
            "Location": "Entertainment Room",
            "Floor": "Basement",
        }
        with (
            patch.object(provider, "_embed_query", return_value=[0.1]),
            patch.object(provider, "_query_vector_db", return_value=[]),
            patch.object(provider, "_get_entity_state", return_value=full),
            patch(EXPOSE, side_effect=lambda hass, assistant, eid: eid == "light.basement"),
        ):
            parsed = json.loads(await provider.get_context("which lights are on?"))

        assert parsed["all_matching_devices"]["by_state"] == {
            "on": {"count": 1, "by_floor": {"Basement": ["Basement Light (Entertainment Room)"]}}
        }


class TestSummarizeDevices:
    """Tests for summarize_devices."""

    def test_groups_by_state_then_floor_with_counts(self):
        """Counts and grouping are done in code."""
        records = [
            {
                "entity_id": "light.a",
                "name": "A",
                "state": "on",
                "area": "Kitchen",
                "floor": "Main",
            },
            {"entity_id": "light.b", "name": "B", "state": "on", "area": "Den", "floor": "Main"},
            {"entity_id": "light.c", "name": "C", "state": "on", "floor": "Upstairs"},
            {"entity_id": "light.d", "name": "D", "state": "off"},
        ]
        assert summarize_devices(records) == {
            "total": 4,
            "by_state": {
                "on": {
                    "count": 3,
                    "by_floor": {"Main": ["A (Kitchen)", "B (Den)"], "Upstairs": ["C"]},
                },
                "off": {"count": 1, "by_floor": {NO_FLOOR: ["D"]}},
            },
        }

    def test_names_go_to_the_smallest_groups_first(self):
        """With a budget, rare states are listed in full and common ones are cut."""
        records = [{"name": f"Off{i}", "state": "off"} for i in range(5)]
        records += [{"name": "On0", "state": "on", "floor": "Main"}]
        assert summarize_devices(records, max_listed=3) == {
            "total": 6,
            "truncated": True,
            "by_state": {
                "off": {"count": 5, "by_floor": {NO_FLOOR: ["Off0", "Off1"]}, "not_listed": 3},
                "on": {"count": 1, "by_floor": {"Main": ["On0"]}},
            },
        }

    def test_budget_that_fits_everything_changes_nothing(self):
        """A budget at or above the number of records lists them all."""
        records = [{"name": "A", "state": "on"}, {"name": "B", "state": "off"}]
        assert summarize_devices(records, max_listed=2) == summarize_devices(records)
        assert "truncated" not in summarize_devices(records, max_listed=2)

    def test_empty(self):
        """No records gives an empty summary."""
        assert summarize_devices([]) == {"total": 0, "by_state": {}}
