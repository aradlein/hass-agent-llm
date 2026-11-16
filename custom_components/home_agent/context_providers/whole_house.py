"""Detect questions about a whole category of devices.

Semantic search returns only the few entities closest to the request. That
is right for "is the office light on?" and wrong for "which lights are
on?", where the answer depends on every light in the home. This module
decides when a request is the second kind and which entities it covers.

Detection is keyword based, with one vocabulary per language in
``_LANGUAGES``. A request in a language without a vocabulary is left alone
and gets the normal semantic-search context. To support another language,
add a ``_Language`` entry; nothing else needs to change.

It has no Home Assistant imports so it can be tested on its own.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EntitySelector:
    """Selects entities by domain, optionally narrowed by device class."""

    domain: str
    device_classes: frozenset[str] | None = None

    def matches(self, domain: str, device_class: str | None) -> bool:
        """Return True if an entity with this domain and device class is selected."""
        if domain != self.domain:
            return False
        return self.device_classes is None or device_class in self.device_classes


def _sel(domain: str, *device_classes: str) -> EntitySelector:
    return EntitySelector(domain, frozenset(device_classes) if device_classes else None)


# Category key -> (label used in the prompt, selectors). Shared by all languages.
_CATEGORY_SELECTORS: dict[str, tuple[str, tuple[EntitySelector, ...]]] = {
    "lights": ("lights", (_sel("light"),)),
    "locks_doors": (
        "locks and doors",
        (
            _sel("lock"),
            _sel("cover", "door", "garage", "gate"),
            _sel("binary_sensor", "door", "garage_door", "opening", "lock"),
        ),
    ),
    "garage": (
        "garage doors",
        (_sel("cover", "garage"), _sel("binary_sensor", "garage_door")),
    ),
    "windows": ("windows", (_sel("binary_sensor", "window"), _sel("cover", "window"))),
    "covers": ("blinds and covers", (_sel("cover"),)),
    "thermostats": ("thermostats", (_sel("climate"),)),
    "fans": ("fans", (_sel("fan"),)),
    "switches": ("switches", (_sel("switch"),)),
    "media": ("media players", (_sel("media_player"),)),
}

# Groups used by "is anything on / open / locked?"
_ON_OFF = ("lights, switches and fans", (_sel("light"), _sel("switch"), _sel("fan")))
_OPENINGS = (
    "doors, windows and covers",
    (_sel("cover"), _sel("binary_sensor", "door", "garage_door", "opening", "window")),
)
_LOCKS = _CATEGORY_SELECTORS["locks_doors"]


def _words(alternatives: str) -> re.Pattern[str]:
    """Compile a whole-word pattern from a '|' separated list."""
    if not alternatives or "" in alternatives.split("|"):
        # An empty alternative would match every request
        raise ValueError(f"Empty alternative in word list: {alternatives!r}")
    return re.compile(rf"\b(?:{alternatives})\b")


@dataclass(frozen=True)
class _Language:
    """Vocabulary for one language.

    All words are written the way ``_normalize`` leaves them: lower case,
    without accents, with punctuation (including apostrophes) as spaces.

    Attributes:
        strong: Words that ask about a whole set whatever the noun's number
            ("which", "any", "all", "how many").
        weak: Question openers that only imply a whole set when the noun is
            plural ("are the doors locked").
        everything: "anything" / "everything".
        on_off, opened, locked: State words that pick the group for an
            "anything / everything" question.
        categories: Category key -> (any form of the noun, plural forms only).
            Where singular and plural are spelled alike (German "Fenster"),
            the plural form includes the plural article ("die fenster").
        command_verbs: First words that make the request a command.
        fillers: Polite lead-ins skipped before looking for a command verb.
        command_anywhere: Verb forms that mark a command wherever they are,
            for languages that put the verb last ("kannst du ... ausschalten").
    """

    strong: str
    weak: str
    everything: str
    on_off: str
    opened: str
    locked: str
    categories: dict[str, tuple[str, str]]
    command_verbs: str
    fillers: str
    command_anywhere: str = ""


_LANGUAGES: dict[str, _Language] = {
    "en": _Language(
        strong="which|any|all|every|each|how many",
        weak="what|are|were|did|have|has",
        everything="anything|everything",
        on_off="on|off",
        opened="open|opened|closed|shut",
        locked="locked|unlocked",
        categories={
            "lights": ("lights?|lamps?", "lights|lamps"),
            "locks_doors": ("locks?|locked|unlocked|doors?", "locks|doors"),
            "garage": ("garages?", "garages"),
            "windows": ("windows?", "windows"),
            "covers": (
                "blinds?|shades?|curtains?|covers?|shutters?",
                "blinds|shades|curtains|covers|shutters",
            ),
            "thermostats": ("thermostats?|hvac", "thermostats"),
            "fans": ("fans?", "fans"),
            "switches": ("switch|switches|plugs?|outlets?", "switches|plugs|outlets"),
            "media": ("speakers?|tvs?|televisions?", "speakers|tvs|televisions"),
        },
        command_verbs=(
            "turn|switch|set|dim|brighten|lock|unlock|open|close|shut|raise|lower|start|stop|"
            "play|pause|make|put|change|increase|decrease|activate|enable|disable|toggle"
        ),
        fillers=(
            "please|can you|could you|would you|will you|go ahead and|"
            "i want you to|i need you to|i would like you to|i d like you to"
        ),
    ),
    "de": _Language(
        strong="welche[rsnm]?|irgendein\\w*|irgendwelche|irgendwo|alle[rsnm]?|jede[rsnm]?|"
        "wie viele|wieviele",
        weak="sind|waren|was|habe|hab|haben",
        everything="alles|irgendwas|irgendetwas|etwas",
        on_off="an|aus|eingeschaltet|ausgeschaltet|angeschaltet",
        opened="offen|geoffnet|auf|zu|geschlossen",
        locked="abgeschlossen|verriegelt|entriegelt|verschlossen|aufgeschlossen|abgesperrt",
        categories={
            "lights": (
                "licht|lichter|lampe|lampen|leuchte|leuchten|beleuchtung",
                "lichter|lampen|leuchten",
            ),
            "locks_doors": (
                "schloss|schlosser|tur|turen|abgeschlossen|verriegelt|entriegelt|"
                "verschlossen|aufgeschlossen|abgesperrt",
                "schlosser|turen",
            ),
            "garage": ("garage|garagen|garagentor|garagentore", "garagen|garagentore"),
            "windows": ("fenster", "die fenster"),
            "covers": (
                "rollo|rollos|rolladen|rollladen|jalousie|jalousien|vorhang|vorhange|"
                "markise|markisen",
                "rollos|rolladen|rollladen|jalousien|vorhange|markisen",
            ),
            "thermostats": (
                "thermostat|thermostate|heizung|heizungen|heizkorper",
                "thermostate|heizungen|die heizkorper",
            ),
            "fans": ("ventilator|ventilatoren|lufter", "ventilatoren|die lufter"),
            "switches": ("schalter|steckdose|steckdosen", "die schalter|steckdosen"),
            "media": (
                "lautsprecher|fernseher|tv|tvs",
                "die lautsprecher|die fernseher|tvs",
            ),
        },
        command_verbs=(
            "schalte|schalt|mach|mache|stell|stelle|dimme|offne|schliesse|schliess|verriegle|"
            "verriegel|entriegle|entriegel|sperre|sperr|starte|stoppe|spiele|spiel|pausiere|"
            "setze|setz|dreh|drehe|aktiviere|deaktiviere|fahre|fahr|lass"
        ),
        fillers="bitte|kannst du|konntest du|konnen sie|konnten sie|wurdest du",
        command_anywhere=(
            "\\w*schalten|an(?:zu)?machen|aus(?:zu)?machen|auf(?:zu)?machen|zu(?:zu)?machen|"
            "offnen|schliessen|abschliessen|verriegeln|entriegeln|stellen|dimmen"
        ),
    ),
    "fr": _Language(
        strong="quel|quelle|quels|quelles|lesquels|lesquelles|tous|toutes|toute|chaque|"
        "combien|aucun|aucune",
        weak="sont|etaient|il y a|y a t il|il reste|reste t il",
        everything="quelque chose|tout|rien",
        on_off="allume\\w*|eteint\\w*|en marche",
        opened="ouvert\\w*|ferme\\w*",
        locked="verrouille\\w*|deverrouille\\w*",
        categories={
            "lights": (
                "lumiere|lumieres|lampe|lampes|eclairage|eclairages",
                "lumieres|lampes|eclairages",
            ),
            "locks_doors": (
                "serrure|serrures|porte|portes|verrou|verrous|verrouille\\w*|deverrouille\\w*",
                "serrures|portes|verrous",
            ),
            "garage": ("garage|garages", "garages"),
            "windows": ("fenetre|fenetres", "fenetres"),
            "covers": ("store|stores|volet|volets|rideau|rideaux", "stores|volets|rideaux"),
            "thermostats": (
                "thermostat|thermostats|chauffage|chauffages|radiateur|radiateurs",
                "thermostats|chauffages|radiateurs",
            ),
            "fans": ("ventilateur|ventilateurs", "ventilateurs"),
            "switches": ("interrupteur|interrupteurs|prise|prises", "interrupteurs|prises"),
            "media": (
                "enceinte|enceintes|haut parleur|haut parleurs|tele|teles|television|"
                "televisions|tv",
                "enceintes|haut parleurs|teles|televisions",
            ),
        },
        command_verbs=(
            "eteins|eteint|eteignez|eteindre|allume|allumez|allumer|ferme|fermez|fermer|"
            "ouvre|ouvrez|ouvrir|verrouille|verrouillez|verrouiller|deverrouille|"
            "deverrouillez|deverrouiller|mets|mettez|mettre|regle|reglez|regler|baisse|"
            "baissez|baisser|monte|montez|monter|arrete|arretez|arreter|lance|lancez|lancer|"
            "active|activez|activer|desactive|desactivez|desactiver|coupe|coupez|couper"
        ),
        fillers=(
            "s il te plait|s il vous plait|peux tu|pouvez vous|tu peux|vous pouvez|"
            "pourrais tu|pourriez vous|merci de|veuillez"
        ),
    ),
    "es": _Language(
        strong="cual|cuales|cuantas|cuantos|todas|todos|cada|algun\\w*|ningun\\w*",
        weak="que|estan|estaban|son|quedan|deje|hay",
        everything="algo|todo|nada",
        on_off="encendid\\w*|apagad\\w*|prendid\\w*",
        opened="abiert\\w*|cerrad\\w*",
        locked="bloquead\\w*|desbloquead\\w*|con llave",
        categories={
            "lights": (
                "luz|luces|lampara|lamparas|foco|focos|bombilla|bombillas",
                "luces|lamparas|focos|bombillas",
            ),
            "locks_doors": (
                "cerradura|cerraduras|puerta|puertas|cerrojo|cerrojos|bloquead\\w*|"
                "desbloquead\\w*",
                "cerraduras|puertas|cerrojos",
            ),
            "garage": ("garaje|garajes|cochera|cocheras", "garajes|cocheras"),
            "windows": ("ventana|ventanas", "ventanas"),
            "covers": (
                "persiana|persianas|cortina|cortinas|toldo|toldos|estor|estores",
                "persianas|cortinas|toldos|estores",
            ),
            "thermostats": (
                "termostato|termostatos|calefaccion|calefacciones",
                "termostatos|calefacciones",
            ),
            "fans": ("ventilador|ventiladores", "ventiladores"),
            "switches": ("interruptor|interruptores|enchufe|enchufes", "interruptores|enchufes"),
            "media": (
                "altavoz|altavoces|bocina|bocinas|tele|teles|television|televisiones|"
                "televisor|televisores|tv",
                "altavoces|bocinas|teles|televisiones|televisores",
            ),
        },
        command_verbs=(
            "apaga|apague|apagar|enciende|encienda|encender|prende|prender|cierra|cierre|"
            "cerrar|abre|abra|abrir|bloquea|bloquear|desbloquea|desbloquear|pon|ponga|poner|"
            "sube|subir|baja|bajar|ajusta|ajustar|activa|activar|desactiva|desactivar|"
            "reproduce|pausa|deten"
        ),
        fillers="por favor|puedes|puede|podrias|podria",
    ),
    "nl": _Language(
        strong="welke|alle|elke|iedere|hoeveel|ergens",
        weak="staan|zijn|waren|heb|branden",
        everything="alles|iets|niets",
        on_off="aan|uit",
        opened="open|dicht|gesloten|geopend",
        locked="vergrendeld|ontgrendeld|op slot|van het slot",
        categories={
            "lights": ("licht|lichten|lamp|lampen|verlichting", "lichten|lampen"),
            "locks_doors": ("slot|sloten|deur|deuren|vergrendeld|ontgrendeld", "sloten|deuren"),
            "garage": ("garage|garages|garagedeur|garagedeuren", "garages|garagedeuren"),
            "windows": ("raam|ramen", "ramen"),
            "covers": (
                "gordijn|gordijnen|rolluik|rolluiken|zonwering|jaloezie|jaloezieen|"
                "screen|screens",
                "gordijnen|rolluiken|jaloezieen|screens",
            ),
            "thermostats": ("thermostaat|thermostaten|verwarming", "thermostaten"),
            "fans": ("ventilator|ventilatoren|ventilators", "ventilatoren|ventilators"),
            "switches": (
                "schakelaar|schakelaars|stopcontact|stopcontacten|stekker|stekkers",
                "schakelaars|stopcontacten|stekkers",
            ),
            "media": (
                "speaker|speakers|luidspreker|luidsprekers|tv|tvs|televisie|televisies",
                "speakers|luidsprekers|tvs|televisies",
            ),
        },
        command_verbs=(
            "zet|doe|schakel|maak|sluit|open|vergrendel|ontgrendel|dim|start|stop|speel|"
            "pauzeer|draai|verlaag|verhoog"
        ),
        fillers="alsjeblieft|alstublieft|kun je|kan je|kunt u|wil je|zou je",
        command_anywhere=(
            "(?:aan|uit)(?:doen|zetten|schakelen)|dichtdoen|opendoen|sluiten|openen|"
            "vergrendelen|ontgrendelen"
        ),
    ),
    "it": _Language(
        strong="quale|quali|quante|quanti|tutte|tutti|ogni|qualche|alcun\\w*|nessun\\w*",
        weak="che|sono|erano|ho|ci sono",
        everything="qualcosa|tutto|niente",
        on_off="acces\\w*|spent\\w*",
        opened="apert\\w*|chius\\w*",
        locked="bloccat\\w*|sbloccat\\w*|a chiave",
        categories={
            "lights": ("luce|luci|lampada|lampade|illuminazione", "luci|lampade"),
            "locks_doors": (
                "serratura|serrature|porta|porte|bloccat\\w*|sbloccat\\w*",
                "serrature|porte",
            ),
            "garage": ("garage|box", "garage"),
            "windows": ("finestra|finestre", "finestre"),
            "covers": (
                "tapparella|tapparelle|tenda|tende|persiana|persiane|serranda|serrande",
                "tapparelle|tende|persiane|serrande",
            ),
            "thermostats": ("termostato|termostati|riscaldamento", "termostati"),
            "fans": ("ventilatore|ventilatori|ventola|ventole", "ventilatori|ventole"),
            "switches": ("interruttore|interruttori|presa|prese", "interruttori|prese"),
            "media": (
                "altoparlante|altoparlanti|cassa|casse|tv|televisore|televisori|"
                "televisione|televisioni",
                "altoparlanti|casse|televisori|televisioni",
            ),
        },
        command_verbs=(
            "spegni|spegnere|spegnete|accendi|accendere|accendete|chiudi|chiudere|apri|"
            "aprire|blocca|bloccare|sblocca|sbloccare|metti|imposta|impostare|alza|abbassa|"
            "attiva|disattiva|ferma|avvia|riproduci"
        ),
        fillers="per favore|per piacere|puoi|potresti|puo",
    ),
}


@dataclass(frozen=True)
class _Compiled:
    """A ``_Language`` with its patterns compiled."""

    strong: re.Pattern[str]
    weak: re.Pattern[str]
    everything: re.Pattern[str]
    on_off: re.Pattern[str]
    opened: re.Pattern[str]
    locked: re.Pattern[str]
    categories: tuple[tuple[str, re.Pattern[str], re.Pattern[str]], ...]


_COMPILED: tuple[_Compiled, ...] = tuple(
    _Compiled(
        strong=_words(lang.strong),
        weak=_words(lang.weak),
        everything=_words(lang.everything),
        on_off=_words(lang.on_off),
        opened=_words(lang.opened),
        locked=_words(lang.locked),
        categories=tuple(
            (key, _words(any_form), _words(plural))
            for key, (any_form, plural) in lang.categories.items()
        ),
    )
    for lang in _LANGUAGES.values()
)

# Command detection looks at every language at once: a request that starts
# with a control verb in any of them is a command.
_CONTROL_VERBS = frozenset(
    verb for lang in _LANGUAGES.values() for verb in lang.command_verbs.split("|")
)
_COMMAND_ANYWHERE = _words(
    "|".join(lang.command_anywhere for lang in _LANGUAGES.values() if lang.command_anywhere)
)
_WAKE_WORD = re.compile(r"^(?:hey|ok|okay|hallo|hola|salut|ciao)\s+\w+\s+")
_LEADING_FILLER = re.compile(
    r"^(?:(?:" + "|".join(lang.fillers for lang in _LANGUAGES.values()) + r")\s+)+"
)


def _normalize(user_input: str) -> str:
    """Lower-case, strip accents and turn punctuation into single spaces."""
    text = unicodedata.normalize("NFKD", user_input.casefold())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^\w ]+|_", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_command(text: str) -> bool:
    rest = _WAKE_WORD.sub("", text, count=1)
    rest = _LEADING_FILLER.sub("", rest, count=1)
    first = rest.split(" ", 1)[0] if rest else ""
    return first in _CONTROL_VERBS or bool(_COMMAND_ANYWHERE.search(rest))


def detect_whole_house_query(user_input: str) -> tuple[list[str], list[EntitySelector]]:
    """Work out whether a request asks about a whole category of devices.

    Args:
        user_input: The request as transcribed, in any supported language.

    Returns:
        (labels, selectors). Both are empty for single-device questions,
        for commands, and for languages without a vocabulary. Labels name
        the categories, for the prompt.
    """
    text = _normalize(user_input)
    if not text or _is_command(text):
        return [], []

    labels: list[str] = []
    selectors: list[EntitySelector] = []

    def add(group: tuple[str, tuple[EntitySelector, ...]]) -> None:
        label, new = group
        if label not in labels:
            labels.append(label)
        for selector in new:
            if selector not in selectors:
                selectors.append(selector)

    # Each language is checked on its own, so a quantifier in one language
    # never combines with a noun from another.
    for lang in _COMPILED:
        strong = bool(lang.strong.search(text))
        weak = bool(lang.weak.search(text))
        for key, any_form, plural in lang.categories:
            if (strong and any_form.search(text)) or (weak and plural.search(text)):
                add(_CATEGORY_SELECTORS[key])

        if lang.everything.search(text):
            if lang.on_off.search(text):
                add(_ON_OFF)
            if lang.opened.search(text):
                add(_OPENINGS)
            if lang.locked.search(text):
                add(_LOCKS)

    return labels, selectors


# binary_sensor reports on/off whatever it measures. Translate it into the
# words the matching covers and locks use, so one question about "open" or
# "locked" things does not have to be answered from two vocabularies.
_BINARY_SENSOR_STATES: dict[str, dict[str, str]] = {
    "door": {"on": "open", "off": "closed"},
    "garage_door": {"on": "open", "off": "closed"},
    "opening": {"on": "open", "off": "closed"},
    "window": {"on": "open", "off": "closed"},
    "lock": {"on": "unlocked", "off": "locked"},
}


def normalize_state(domain: str, device_class: str | None, state: str) -> str:
    """Return the state in plain words for the device kind.

    Args:
        domain: Entity domain.
        device_class: Device class, if any.
        state: Raw Home Assistant state.

    Returns:
        "open"/"closed" or "unlocked"/"locked" for the binary sensors that
        mean that; the state unchanged for everything else.
    """
    if domain != "binary_sensor" or device_class is None:
        return state
    return _BINARY_SENSOR_STATES.get(device_class, {}).get(state, state)


NO_FLOOR = "Floor not set"


def summarize_devices(
    records: list[dict[str, Any]], max_listed: int | None = None
) -> dict[str, Any]:
    """Group device records by state and floor so the answer can be read off.

    Small models miscount and drop items when asked to filter a long flat
    list, so the filtering and counting is done here instead.

    Counts always cover every record. When there are more records than
    ``max_listed``, names are listed for the smallest state groups first
    (the lights that are on, the door that is unlocked) and the remainder
    of a group is reported as ``not_listed``, so the model knows a list is
    partial instead of treating it as complete.

    Args:
        records: Dicts with "name" and "state", optionally "area", "floor"
            and any extra attributes worth mentioning.
        max_listed: Most device names to list across all groups. None lists all.

    Returns:
        {"total": n, "by_state": {state: {"count": n, "by_floor": {floor: [text]}}}}
        plus "truncated": True, and "not_listed" on the affected groups,
        when names were left out.
    """
    grouped: dict[str, list[tuple[str, str]]] = {}
    for record in records:
        text = str(record["name"])
        if record.get("area"):
            text += f" ({record['area']})"
        extras = [
            f"{key.replace('_', ' ')} {value}"
            for key, value in record.items()
            if key not in ("entity_id", "name", "state", "area", "floor") and value is not None
        ]
        if extras:
            text += ", " + ", ".join(extras)
        grouped.setdefault(str(record["state"]), []).append((record.get("floor") or NO_FLOOR, text))

    # Hand out the name budget to the smallest groups first
    budget = len(records) if max_listed is None else max(max_listed, 0)
    listed: dict[str, int] = {}
    for state in sorted(grouped, key=lambda s: len(grouped[s])):
        listed[state] = min(len(grouped[state]), budget)
        budget -= listed[state]

    by_state: dict[str, dict[str, Any]] = {}
    truncated = False
    for state, items in grouped.items():
        group: dict[str, Any] = {"count": len(items), "by_floor": {}}
        for floor, text in items[: listed[state]]:
            group["by_floor"].setdefault(floor, []).append(text)
        if listed[state] < len(items):
            group["not_listed"] = len(items) - listed[state]
            truncated = True
        by_state[state] = group

    summary: dict[str, Any] = {"total": len(records), "by_state": by_state}
    if truncated:
        summary["truncated"] = True
    return summary
