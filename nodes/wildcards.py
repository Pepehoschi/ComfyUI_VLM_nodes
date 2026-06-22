import json
import random
import re
import zlib
from pathlib import Path

import folder_paths

from .suggest import (
    _clean_llama_text,
    _create_llama_text_response,
    _llama_text_response,
)


WILDCARD_TOKEN_RE = re.compile(r"__([A-Za-z0-9_./\\-]+)__")
WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]+")
IGNORED_WILDCARD_FILENAMES = {
    "changelog.txt",
    "license.txt",
    "readme.txt",
}
SCHOOL_EVIDENCE = {
    "academy",
    "campus",
    "class",
    "classroom",
    "club",
    "school",
    "student",
    "teacher",
    "uniform",
}
MAGIC_EVIDENCE = {
    "fantasy",
    "magic",
    "mage",
    "robe",
    "spell",
    "spellbook",
    "staff",
    "wand",
    "wizard",
    "witch",
}
WATER_OUTFIT_CLUES = {
    "beach towel",
    "beachwear",
    "bikini",
    "board shorts",
    "flip flops",
    "flip-flops",
    "goggles",
    "inflatable ring",
    "one piece swimsuit",
    "one-piece swimsuit",
    "rash guard",
    "sandals",
    "sarong",
    "school swimsuit",
    "sundress",
    "swim cap",
    "swim trunks",
    "swimsuit",
    "swimwear",
    "water gun",
    "wetsuit",
}
WATER_CONTEXT_CLUES = {
    "aquatic",
    "bath",
    "beach",
    "hot spring",
    "ocean",
    "onsen",
    "pool",
    "resort",
    "sea",
    "summer",
    "tropical",
    "water park",
    "waterpark",
}
WATER_WORLD_TAGS = {
    "fantasy",
    "isekai",
    "magic",
    "adventure",
    "sci fi",
    "space",
    "mecha",
}
SUMMER_WATER_FILE_METADATA = {
    "summer_beach_water_locations.txt": {
        "strong_tags": ["Romance", "Slice of Life", "Comedy"],
        "weak_tags": ["School", "Harem", "Adventure"],
        "prompt_clues": ["beach", "summer", "ocean", "sea", "shore", "water fun"],
    },
    "pool_waterpark_locations.txt": {
        "strong_tags": ["School", "Comedy", "Slice of Life"],
        "weak_tags": ["Romance", "Sports", "Harem"],
        "prompt_clues": ["pool", "waterpark", "water park", "school swimsuit", "swim cap", "goggles", "water slide"],
    },
    "hot_spring_resort_locations.txt": {
        "strong_tags": ["Romance", "Slice of Life", "Drama"],
        "weak_tags": ["Comedy", "Harem", "Josei"],
        "outfit_clues": ["yukata", "bath towel", "robe"],
        "prompt_clues": ["hot spring", "onsen", "bath", "resort", "ryokan", "open-air bath"],
    },
    "fantasy_tropical_water_locations.txt": {
        "strong_tags": ["Fantasy", "Adventure", "Magic"],
        "weak_tags": ["Romance", "Comedy", "Harem"],
        "prompt_clues": ["tropical", "lagoon", "island", "beach", "waterfall", "fantasy resort"],
    },
    "sci_fi_aquatic_resort_locations.txt": {
        "strong_tags": ["Sci-Fi", "Space", "Mecha"],
        "weak_tags": ["Romance", "Adventure", "Seinen"],
        "prompt_clues": ["aquatic", "underwater", "resort", "dome", "space beach", "futuristic pool"],
    },
}
OUTFIT_OVERRIDE_FILE_METADATA = {
    "ancient_egyptian_locations.txt": {
        "strong_outfit_clues": [
            "ancient egyptian outfit",
            "egyptian outfit",
            "pharaoh outfit",
            "cleopatra outfit",
            "ankh jewelry",
            "gold collar",
            "nemes headdress",
            "linen wrap dress",
            "egyptian priestess",
            "hieroglyph accessories",
            "scarab ornament",
        ],
        "strong_tags": ["Historical", "Fantasy", "Adventure"],
        "weak_tags": ["Magic", "Romance", "Drama"],
        "negative_clues": ["school uniform", "spacesuit", "modern idol outfit"],
    },
    "pirate_locations.txt": {
        "strong_outfit_clues": [
            "pirate outfit",
            "tricorn hat",
            "captain coat",
            "eyepatch",
            "cutlass",
            "sailor pirate",
            "corsair outfit",
            "buccaneer outfit",
            "skull emblem",
            "treasure map",
            "naval rogue",
        ],
        "strong_tags": ["Pirate", "Adventure", "Fantasy", "Historical", "Action"],
        "weak_tags": ["Comedy", "Drama", "Romance"],
        "negative_clues": ["school uniform", "space suit", "office suit"],
    },
    "full_armor_fantasy_locations.txt": {
        "strong_outfit_clues": [
            "full armor",
            "plate armor",
            "knight armor",
            "paladin armor",
            "fantasy armor",
            "armored girl",
            "breastplate",
            "gauntlets",
            "pauldrons",
            "helmet",
            "sword and shield",
            "medieval armor",
        ],
        "strong_tags": ["Fantasy", "Adventure", "Action", "Magic"],
        "weak_tags": ["Shounen", "Seinen", "Drama", "Romance"],
        "negative_clues": ["mecha pilot suit", "power armor", "sci-fi armor"],
    },
    "desert_fantasy_locations.txt": {
        "strong_outfit_clues": [
            "desert robe",
            "belly dancer inspired fantasy outfit",
            "sand mage outfit",
            "turban",
            "veiled dancer",
            "desert princess",
            "gold desert jewelry",
            "oasis traveler",
            "caravan outfit",
        ],
        "strong_tags": ["Fantasy", "Adventure"],
        "weak_tags": ["Historical", "Magic", "Romance"],
        "negative_clues": ["modern bikini", "school uniform", "spacesuit"],
    },
    "naval_adventure_locations.txt": {
        "strong_outfit_clues": [
            "sailor uniform not school",
            "naval coat",
            "naval captain",
            "captain uniform",
            "maritime outfit",
            "ship crew outfit",
            "admiral coat",
            "sailor cap",
            "anchor emblem",
        ],
        "strong_tags": ["Adventure", "Military", "Historical"],
        "weak_tags": ["Action", "Drama", "Romance"],
        "negative_clues": ["school sailor uniform", "space suit", "idol sailor costume"],
    },
}
FILENAME_FALLBACK_METADATA = {
    **SUMMER_WATER_FILE_METADATA,
    **OUTFIT_OVERRIDE_FILE_METADATA,
}
SERIES_TAG_ALIASES = {
    "action": ["Action"],
    "adventure": ["Adventure"],
    "alien": ["Sci-Fi", "Space"],
    "aliens": ["Sci-Fi", "Space"],
    "aliens living on earth": ["Sci-Fi"],
    "alternative past": ["Historical"],
    "alternative present": ["Contemporary"],
    "anachronism": ["Historical"],
    "android": ["Sci-Fi"],
    "androids": ["Sci-Fi"],
    "artificial intelligence": ["Sci-Fi"],
    "bakumatsu meiji period": ["Historical", "Samurai"],
    "bakumatsu meiji": ["Historical", "Samurai"],
    "bakumatsu": ["Historical", "Samurai"],
    "bar": ["Contemporary", "City"],
    "badminton": ["Sports"],
    "bounty hunter": ["Action", "Sci-Fi"],
    "boxing": ["Sports", "Martial Arts"],
    "clans": ["Historical", "Fantasy"],
    "comedy": ["Comedy"],
    "contemporary fantasy": ["Fantasy", "Supernatural", "Contemporary"],
    "crime": ["Police", "Mystery", "City"],
    "daily life": ["Slice of Life", "Contemporary"],
    "death game": ["Game", "Thriller"],
    "detective": ["Police", "Mystery"],
    "drama": ["Drama"],
    "earth": ["Contemporary"],
    "ecchi": ["Romance"],
    "edo period": ["Historical", "Samurai"],
    "fantasy": ["Fantasy"],
    "feudal japan": ["Historical", "Samurai"],
    "fishing": ["Water", "Slice of Life"],
    "future": ["Sci-Fi"],
    "futuristic city": ["Sci-Fi", "City"],
    "futuristic school": ["Sci-Fi", "School"],
    "gangs": ["City", "Action"],
    "ghost": ["Supernatural", "Horror"],
    "gore": ["Horror"],
    "greek mythology": ["Fantasy", "Historical"],
    "historical": ["Historical"],
    "japan": ["Contemporary"],
    "law and order": ["Police"],
    "mafia": ["City", "Police"],
    "magic": ["Magic", "Fantasy"],
    "magical": ["Magic", "Fantasy"],
    "martial arts": ["Martial Arts"],
    "mecha": ["Mecha", "Sci-Fi"],
    "military": ["Military"],
    "music": ["Music"],
    "musical band": ["Music"],
    "mystery": ["Mystery"],
    "ninja": ["Historical", "Samurai", "Martial Arts"],
    "old asia": ["Historical"],
    "parallel universe": ["Fantasy", "Sci-Fi"],
    "parallel world": ["Fantasy"],
    "parody": ["Parody"],
    "past": ["Historical"],
    "pirate": ["Pirate", "Adventure"],
    "pirates": ["Pirate", "Adventure"],
    "place": [],
    "police": ["Police"],
    "present": ["Contemporary"],
    "psi powers": ["Super Power", "Sci-Fi"],
    "psi-powers": ["Super Power", "Sci-Fi"],
    "robot": ["Sci-Fi", "Mecha"],
    "robots": ["Sci-Fi", "Mecha"],
    "samurai": ["Samurai", "Historical"],
    "school": ["School"],
    "school life": ["School", "Slice of Life"],
    "sci fi": ["Sci-Fi"],
    "sci-fi": ["Sci-Fi"],
    "science and magic coexist": ["Sci-Fi", "Magic", "Fantasy"],
    "science fiction": ["Sci-Fi"],
    "science-fiction": ["Sci-Fi"],
    "shinsengumi": ["Historical", "Samurai", "Police"],
    "skateboarding": ["Sports", "Contemporary"],
    "space": ["Space", "Sci-Fi"],
    "space travel": ["Space", "Sci-Fi"],
    "special squads": ["Military", "Police", "Action"],
    "speculative fiction": ["Sci-Fi", "Fantasy"],
    "sports": ["Sports"],
    "high school": ["School"],
    "slice of life drama": ["Slice of Life", "Drama"],
    "super power": ["Super Power"],
    "superpowers": ["Super Power"],
    "supernatural": ["Supernatural"],
    "swordplay": ["Action", "Fantasy", "Samurai"],
    "swords co": ["Action", "Fantasy"],
    "the arts": ["Music"],
    "thriller": ["Thriller"],
    "urban": ["City", "Contemporary"],
    "urban fantasy": ["Fantasy", "Supernatural", "City", "Contemporary"],
    "vocaloid": ["Music"],
    "yakuza": ["City", "Police"],
}
SERIES_TAG_CANONICALS = {
    "Action",
    "Adventure",
    "City",
    "Comedy",
    "Contemporary",
    "Drama",
    "Fantasy",
    "Game",
    "Historical",
    "Horror",
    "Magic",
    "Martial Arts",
    "Mecha",
    "Military",
    "Music",
    "Mystery",
    "Parody",
    "Pirate",
    "Police",
    "Romance",
    "Samurai",
    "School",
    "Sci-Fi",
    "Slice of Life",
    "Space",
    "Sports",
    "Super Power",
    "Supernatural",
    "Thriller",
    "Water",
}


def _as_int(value, default, min_value=None, max_value=None):
    try:
        value = int(value)
    except (TypeError, ValueError):
        return default
    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


def _as_float(value, default, min_value=None, max_value=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


def _normalize_token_name(name):
    name = str(name).strip().replace("\\", "/")
    name = name[:-4] if name.lower().endswith(".txt") else name
    name = name.strip("_")
    return name


def _token(name):
    return f"__{_normalize_token_name(name)}__"


def _normalize_label(value):
    value = str(value or "").casefold()
    value = re.sub(r"[^0-9a-z]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _as_string_list(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    return [str(value).strip()] if str(value).strip() else []


def _merge_unique(existing, extra):
    merged = []
    seen = set()
    for value in [*_as_string_list(existing), *_as_string_list(extra)]:
        normalized = _normalize_label(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            merged.append(value)
    return merged


def _extract_genres(genres):
    text = str(genres or "").strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    values = []

    def collect(value):
        if isinstance(value, str):
            values.extend(part.strip() for part in re.split(r"[,;\n]+", value) if part.strip())
        elif isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            for key in ("genres", "genre", "tags"):
                if key in value:
                    collect(value[key])

    collect(parsed if parsed is not None else text)
    deduped = []
    seen = set()
    for value in values:
        normalized = _normalize_label(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(value)
    return deduped


def _extract_series_routes(series_tags):
    raw_tags = _extract_genres(series_tags)
    routes = []
    evidence = []
    seen = set()
    canonical_by_label = {
        _normalize_label(canonical): canonical
        for canonical in SERIES_TAG_CANONICALS
    }
    for raw_tag in raw_tags:
        normalized = _normalize_label(raw_tag)
        mapped = SERIES_TAG_ALIASES.get(normalized)
        if mapped is None:
            mapped = [canonical_by_label[normalized]] if normalized in canonical_by_label else []
        mapped = [route for route in mapped if route]
        if mapped:
            evidence.append({"tag": raw_tag, "routes": mapped})
        for route in mapped:
            route_key = _normalize_label(route)
            if route_key not in seen:
                seen.add(route_key)
                routes.append(route)
    return routes, evidence


def _contains_normalized(haystack, needle):
    haystack = f" {_normalize_label(haystack)} "
    needle = _normalize_label(needle)
    return bool(needle) and f" {needle} " in haystack


def _manifest_score_terms(values, tags):
    normalized_values = {_normalize_label(value) for value in values}
    return [
        tag for tag in _as_string_list(tags)
        if _normalize_label(tag) in normalized_values
    ]


def _entry_name_route_matches(values, entry):
    name = entry.get("name", "")
    matches = []
    for value in values:
        normalized = _normalize_label(value)
        if not normalized:
            continue
        if _contains_normalized(name, normalized):
            matches.append(value)
            continue
        if normalized == "sci fi" and ("sci_fi" in name or "scifi" in name):
            matches.append(value)
        elif normalized == "slice of life" and ("daily_life" in name or "home" in name):
            matches.append(value)
        elif normalized == "contemporary" and (
            "contemporary" in name or "city" in name or "daily_life" in name or "home" in name
        ):
            matches.append(value)
        elif normalized == "city" and ("city" in name or "cyberpunk" in name):
            matches.append(value)
        elif normalized == "water" and any(part in name for part in ("water", "pool", "beach", "aquatic", "spring")):
            matches.append(value)
    return _merge_unique([], matches)


def _prompt_matches(prompt, clues):
    return [
        clue for clue in _as_string_list(clues)
        if _contains_normalized(prompt, clue)
    ]


def _has_school_evidence(prompt, genres, entry):
    evidence_text = " ".join([prompt, *genres])
    return any(_contains_normalized(evidence_text, term) for term in SCHOOL_EVIDENCE)


def _has_magic_evidence(prompt, genres, entry):
    evidence_text = " ".join([prompt, *genres])
    return any(_contains_normalized(evidence_text, term) for term in MAGIC_EVIDENCE)


def _has_any_clue(text, clues):
    return any(_contains_normalized(text, clue) for clue in clues)


def _has_water_outfit(prompt):
    return _has_any_clue(prompt, WATER_OUTFIT_CLUES)


def _has_water_context(prompt):
    return _has_any_clue(prompt, WATER_CONTEXT_CLUES)


def _has_world_genre(genres, values):
    normalized = {_normalize_label(genre) for genre in genres}
    return any(_normalize_label(value) in normalized for value in values)


def _water_routing_score(prompt, genres, entry):
    name = entry.get("name", "")
    if name not in {_normalize_token_name(filename) for filename in SUMMER_WATER_FILE_METADATA}:
        return 0, []

    evidence = []
    has_swimwear = _has_water_outfit(prompt)
    has_water_context = _has_water_context(prompt)
    has_yukata_onsen = _contains_normalized(prompt, "yukata") and (
        _contains_normalized(prompt, "hot spring")
        or _contains_normalized(prompt, "onsen")
        or _contains_normalized(prompt, "bath")
        or _contains_normalized(prompt, "resort")
    )
    if has_swimwear:
        evidence.append("swimwear/water outfit")
    if has_water_context:
        evidence.append("water/summer context")
    if has_yukata_onsen:
        evidence.append("yukata + hot spring/resort")
    if not evidence:
        return 0, []

    fantasy_world = _has_world_genre(genres, {"Fantasy", "Magic", "Adventure", "Isekai"})
    scifi_world = _has_world_genre(genres, {"Sci-Fi", "Space", "Mecha"})
    school_world = _has_world_genre(genres, {"School"})
    romance_daily = _has_world_genre(genres, {"Romance", "Slice of Life", "Comedy", "Harem"})

    score = 100 if has_swimwear else 70
    if has_water_context:
        score += 30
    if has_yukata_onsen:
        score += 90

    if name == "pool_waterpark_locations":
        if school_world:
            score += 70
            evidence.append("School -> pool/waterpark")
        if _contains_normalized(prompt, "school swimsuit") or _contains_normalized(prompt, "pool"):
            score += 35
    elif name == "summer_beach_water_locations":
        if romance_daily or not (fantasy_world or scifi_world):
            score += 50
            evidence.append("daily/romance/default summer")
        if _contains_normalized(prompt, "beach") or _contains_normalized(prompt, "bikini"):
            score += 35
    elif name == "hot_spring_resort_locations":
        if has_yukata_onsen:
            score += 80
        elif romance_daily and has_water_context:
            score += 30
    elif name == "fantasy_tropical_water_locations":
        if fantasy_world:
            score += 85
            evidence.append("Fantasy/Magic/Adventure -> tropical water")
        else:
            score -= 60
    elif name == "sci_fi_aquatic_resort_locations":
        if scifi_world:
            score += 85
            evidence.append("Sci-Fi/Space/Mecha -> aquatic resort")
        else:
            score -= 70

    return score, evidence


def _candidate_roots(root_text):
    roots = []
    for raw in str(root_text or "").splitlines():
        raw = raw.strip().strip('"')
        if not raw:
            continue
        path = Path(raw)
        if not path.is_absolute():
            path = Path(folder_paths.base_path) / path
        roots.append(path)

    if roots:
        return roots

    base = Path(folder_paths.base_path)
    custom_nodes = base / "custom_nodes"
    defaults = [
        base / "wildcards",
        base / "user" / "wildcards",
        base / "custom_nodes" / "ComfyUI-Impact-Pack" / "wildcards",
        base / "custom_nodes" / "comfyui_sageutils" / "wildcards",
    ]
    if custom_nodes.exists():
        defaults.extend(path for path in custom_nodes.glob("*/wildcards") if path.is_dir())
    return defaults


def _load_manifest_metadata(root):
    metadata = {}
    manifests = []
    for manifest_path in sorted(root.rglob("manifest.json")):
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        files = data.get("files", {})
        if not isinstance(files, dict):
            continue
        manifests.append(str(manifest_path))
        for filename, raw_meta in files.items():
            if not isinstance(raw_meta, dict):
                continue
            path = (manifest_path.parent / str(filename)).resolve()
            meta = {
                "manifest_path": str(manifest_path),
                "manifest_key": str(filename),
                "strong_tags": _as_string_list(raw_meta.get("strong_tags")),
                "weak_tags": _as_string_list(raw_meta.get("weak_tags")),
                "exclude_tags": _as_string_list(raw_meta.get("exclude_tags")),
                "outfit_clues": _as_string_list(raw_meta.get("outfit_clues")),
                "strong_outfit_clues": _as_string_list(raw_meta.get("strong_outfit_clues")),
                "prompt_clues": _as_string_list(raw_meta.get("prompt_clues")),
                "negative_clues": _as_string_list(raw_meta.get("negative_clues")),
            }
            metadata[str(path).lower()] = meta
    return metadata, manifests


def _patch_file_metadata(filename, raw_meta=None):
    file_meta = FILENAME_FALLBACK_METADATA.get(Path(filename).name, {})
    raw_meta = raw_meta if isinstance(raw_meta, dict) else {}
    return {
        "strong_tags": _merge_unique(file_meta.get("strong_tags"), raw_meta.get("strong_tags")),
        "weak_tags": _merge_unique(file_meta.get("weak_tags"), raw_meta.get("weak_tags")),
        "exclude_tags": _merge_unique(file_meta.get("exclude_tags"), raw_meta.get("exclude_tags")),
        "outfit_clues": _merge_unique(file_meta.get("outfit_clues"), raw_meta.get("outfit_clues")),
        "strong_outfit_clues": _merge_unique(
            file_meta.get("strong_outfit_clues"),
            raw_meta.get("strong_outfit_clues"),
        ),
        "prompt_clues": _merge_unique(file_meta.get("prompt_clues"), raw_meta.get("prompt_clues")),
        "negative_clues": _merge_unique(file_meta.get("negative_clues"), raw_meta.get("negative_clues")),
    }


def _load_routing_patch_metadata(root):
    metadata = {}
    patches = []
    for patch_path in sorted(root.rglob("*routing_patch.json")):
        try:
            data = json.loads(patch_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        patches.append(str(patch_path))
        common_outfits = _as_string_list(data.get("outfit_clues"))
        negative_clues = _as_string_list(data.get("negative_clues"))
        files = data.get("files", {})
        if isinstance(files, dict):
            for filename, raw_meta in files.items():
                if not isinstance(raw_meta, dict):
                    continue
                path = (patch_path.parent / filename).resolve()
                file_meta = _patch_file_metadata(filename, raw_meta)
                metadata[str(path).lower()] = {
                    "manifest_path": str(patch_path),
                    "manifest_key": filename,
                    "routing_priority": "hard_outfit",
                    "strong_tags": file_meta.get("strong_tags", []),
                    "weak_tags": file_meta.get("weak_tags", []),
                    "exclude_tags": file_meta.get("exclude_tags", []),
                    "outfit_clues": _merge_unique(common_outfits, file_meta.get("outfit_clues")),
                    "strong_outfit_clues": file_meta.get("strong_outfit_clues", []),
                    "prompt_clues": file_meta.get("prompt_clues", []),
                    "negative_clues": _merge_unique(negative_clues, file_meta.get("negative_clues")),
                    "routing_conditions": [],
                }

        routing_rules = data.get("routing_rules", [])
        if not isinstance(routing_rules, list):
            continue
        for rule in routing_rules:
            if not isinstance(rule, dict):
                continue
            condition = str(rule.get("condition") or "")
            files = _as_string_list(rule.get("files"))
            for filename in files:
                path = (patch_path.parent / filename).resolve()
                file_meta = _patch_file_metadata(filename)
                current = metadata.setdefault(str(path).lower(), {
                    "manifest_path": str(patch_path),
                    "manifest_key": filename,
                    "routing_priority": "outfit_first",
                    "strong_tags": [],
                    "weak_tags": [],
                    "exclude_tags": [],
                    "outfit_clues": [],
                    "strong_outfit_clues": [],
                    "prompt_clues": [],
                    "negative_clues": negative_clues,
                    "routing_conditions": [],
                })
                current["strong_tags"] = _merge_unique(current.get("strong_tags"), file_meta.get("strong_tags"))
                current["weak_tags"] = _merge_unique(current.get("weak_tags"), file_meta.get("weak_tags"))
                current["strong_outfit_clues"] = _merge_unique(
                    current.get("strong_outfit_clues"),
                    file_meta.get("strong_outfit_clues"),
                )
                current["outfit_clues"] = _merge_unique(
                    current.get("outfit_clues"),
                    [*common_outfits, *(_as_string_list(file_meta.get("outfit_clues")))],
                )
                current["prompt_clues"] = _merge_unique(current.get("prompt_clues"), file_meta.get("prompt_clues"))
                current["negative_clues"] = _merge_unique(
                    current.get("negative_clues"),
                    file_meta.get("negative_clues"),
                )
                current["routing_conditions"] = _merge_unique(current.get("routing_conditions"), [condition])
    return metadata, patches


def _merge_manifest_metadata(base, extra):
    merged = dict(base or {})
    for key in (
        "strong_tags",
        "weak_tags",
        "exclude_tags",
        "outfit_clues",
        "strong_outfit_clues",
        "prompt_clues",
        "negative_clues",
        "routing_conditions",
    ):
        merged[key] = _merge_unique(merged.get(key), extra.get(key))
    for key, value in extra.items():
        if key not in merged or not merged.get(key):
            merged[key] = value
    return merged


def _read_wildcard_lines(path, max_lines=1000):
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = path.read_text(encoding="cp1252", errors="replace")
        except OSError:
            return []
    except OSError:
        return []

    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        lines.append(line)
        if max_lines is not None and len(lines) >= max_lines:
            break
    return lines


def _sample_wildcard_lines(lines, sample_lines, sample_mode, sample_seed, path):
    sample_lines = _as_int(sample_lines, 12, 0)
    if sample_lines <= 0 or not lines:
        return []
    if sample_lines >= len(lines):
        return list(lines)

    sample_mode = str(sample_mode or "first").strip().lower()
    if sample_mode == "random":
        path_seed = zlib.crc32(str(path).replace("\\", "/").lower().encode("utf-8"))
        rng = random.Random(_as_int(sample_seed, 0) + path_seed)
        return rng.sample(lines, sample_lines)
    if sample_mode == "evenly_spaced":
        if sample_lines == 1:
            return [lines[0]]
        last_index = len(lines) - 1
        indexes = [
            round(index * last_index / (sample_lines - 1))
            for index in range(sample_lines)
        ]
        return [lines[index] for index in indexes]
    return lines[:sample_lines]


def build_catalog(root_text, max_files=500, sample_lines=12, sample_mode="first", sample_seed=0):
    catalog = {
        "roots": [],
        "manifests": [],
        "routing_patches": [],
        "tokens": {},
    }
    seen_paths = set()
    for root in _candidate_roots(root_text):
        if not root.exists() or not root.is_dir():
            continue
        manifest_metadata, manifests = _load_manifest_metadata(root)
        patch_metadata, patches = _load_routing_patch_metadata(root)
        catalog["roots"].append(str(root))
        catalog["manifests"].extend(manifests)
        catalog["routing_patches"].extend(patches)
        for path in sorted(root.rglob("*.txt")):
            if path.name.casefold() in IGNORED_WILDCARD_FILENAMES:
                continue
            if len(catalog["tokens"]) >= max_files:
                break
            resolved = str(path.resolve()).lower()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            rel = path.relative_to(root).with_suffix("")
            name = _normalize_token_name(rel.as_posix())
            token = _token(name)
            lines = _read_wildcard_lines(path, max_lines=None)
            entry = {
                "name": name,
                "path": str(path),
                "line_count": len(lines),
                "samples": _sample_wildcard_lines(lines, sample_lines, sample_mode, sample_seed, path),
            }
            if resolved in manifest_metadata:
                entry["manifest"] = manifest_metadata[resolved]
            if resolved in patch_metadata:
                entry["manifest"] = _merge_manifest_metadata(entry.get("manifest"), patch_metadata[resolved])
            filename_meta = FILENAME_FALLBACK_METADATA.get(path.name)
            if filename_meta:
                entry["manifest"] = _merge_manifest_metadata(entry.get("manifest"), {
                    "manifest_path": "built-in filename fallback",
                    "manifest_key": path.name,
                    "routing_priority": "filename_fallback",
                    **_patch_file_metadata(path.name),
                })
            catalog["tokens"][token] = entry
    return catalog


def _split_choice_options(text):
    options = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        if char == "{":
            depth += 1
        elif char == "}":
            depth = max(0, depth - 1)
        elif char == "|" and depth == 0:
            options.append(text[start:index])
            start = index + 1
    options.append(text[start:])
    return options


def expand_choices(text, rng, max_depth=20):
    text = str(text)
    for _ in range(max_depth):
        end = text.find("}")
        if end < 0:
            return text
        start = text.rfind("{", 0, end)
        if start < 0:
            return text
        options = _split_choice_options(text[start + 1:end])
        choice = rng.choice(options) if options else ""
        text = text[:start] + choice + text[end + 1:]
    return text


def resolve_token(catalog, token, rng):
    entry = catalog.get("tokens", {}).get(token)
    if not entry:
        return None
    lines = _read_wildcard_lines(Path(entry["path"]))
    if not lines:
        return ""
    return expand_choices(rng.choice(lines), rng)


def resolve_text(catalog, text, seed, missing_mode="keep", max_depth=20):
    rng = random.Random(_as_int(seed, 0))
    used = []

    def replace(match):
        token = _token(match.group(1))
        resolved = resolve_token(catalog, token, rng)
        if resolved is None:
            if missing_mode == "error":
                raise ValueError(f"Wildcard not found: {token}")
            return "" if missing_mode == "remove" else match.group(0)
        used.append({"token": token, "value": resolved})
        return resolved

    resolved = str(text or "")
    for _ in range(max_depth):
        previous = resolved
        resolved = WILDCARD_TOKEN_RE.sub(replace, resolved)
        resolved = expand_choices(resolved, rng)
        if resolved == previous or (not WILDCARD_TOKEN_RE.search(resolved) and "{" not in resolved):
            break
    return resolved, used


def _score_entry(prompt, category_hint, token, entry):
    haystack = " ".join(
        [token, entry.get("name", "")] + entry.get("samples", [])[:8]
    ).lower()
    words = {
        word for word in WORD_RE.findall(prompt.lower())
        if len(word) > 2
    }
    matches = [word for word in words if word in haystack]
    score = len(matches)
    hint_matches = []
    for hint in WORD_RE.findall(category_hint.lower()):
        if hint in haystack:
            score += 8
            hint_matches.append(hint)
    return score, {
        "text_matches": matches[:20],
        "category_matches": hint_matches,
    }


def _score_manifest_entry(prompt, genres, series_routes, category_hint, token, entry):
    meta = entry.get("manifest") or {}
    strong = _manifest_score_terms(genres, meta.get("strong_tags"))
    weak = _manifest_score_terms(genres, meta.get("weak_tags"))
    excluded = _manifest_score_terms(genres, meta.get("exclude_tags"))
    series_strong = _manifest_score_terms(series_routes, meta.get("strong_tags"))
    series_weak = _manifest_score_terms(series_routes, meta.get("weak_tags"))
    series_name = _entry_name_route_matches(series_routes, entry)
    outfit = _prompt_matches(prompt, meta.get("outfit_clues"))
    strong_outfit = _prompt_matches(prompt, meta.get("strong_outfit_clues"))
    prompt_clues = _prompt_matches(prompt, meta.get("prompt_clues"))
    negative = _prompt_matches(prompt, meta.get("negative_clues"))
    category = [
        hint for hint in WORD_RE.findall(str(category_hint or "").casefold())
        if _contains_normalized(entry.get("name", ""), hint)
    ]

    score = (40 * len(strong)) + (14 * len(weak)) + (26 * len(outfit))
    score += (45 * len(series_strong)) + (18 * len(series_weak)) + (18 * len(series_name))
    score += 100 * len(strong_outfit)
    score += 18 * len(prompt_clues)
    score += 8 * len(category)
    score -= 100 * len(excluded)
    score -= 80 * len(negative)

    water_score, water_evidence = _water_routing_score(prompt, [*genres, *series_routes], entry)
    score += water_score

    name = entry.get("name", "")
    route_context = [*genres, *series_routes]
    schoolish = "school" in name or "academy" in name
    if schoolish and not _has_school_evidence(prompt, route_context, entry):
        score -= 55
    if "magic_academy" in name and not _has_magic_evidence(prompt, route_context, entry):
        score -= 35

    return score, {
        "strong_genre_matches": strong,
        "weak_genre_matches": weak,
        "series_strong_matches": series_strong,
        "series_weak_matches": series_weak,
        "series_name_matches": series_name,
        "outfit_matches": outfit,
        "strong_outfit_matches": strong_outfit,
        "prompt_clue_matches": prompt_clues,
        "exclude_matches": excluded,
        "negative_clue_matches": negative,
        "category_matches": category,
        "water_priority_matches": water_evidence,
        "school_evidence": _has_school_evidence(prompt, route_context, entry),
    }


def _candidate_catalog(catalog, prompt, category_hint, max_candidates, genres="", manifest_mode="auto", series_tags=""):
    all_items = []
    manifest_items = []
    parsed_genres = _extract_genres(genres)
    series_routes, series_evidence = _extract_series_routes(series_tags)
    for token, entry in catalog.get("tokens", {}).items():
        if entry.get("manifest"):
            score, reasons = _score_manifest_entry(prompt, parsed_genres, series_routes, category_hint, token, entry)
            source = "manifest"
        else:
            score, reasons = _score_entry(prompt, category_hint, token, entry)
            source = "filename_samples"
        item = (score, token, entry, source, reasons)
        all_items.append(item)
        if source == "manifest":
            manifest_items.append(item)

    manifest_mode = str(manifest_mode or "auto").strip().lower()
    if manifest_mode == "manifest_only":
        items = manifest_items
    elif manifest_mode == "all_files":
        items = all_items
    elif manifest_items:
        items = manifest_items
    else:
        items = all_items

    items.sort(key=lambda item: (-item[0], item[1]))
    if manifest_mode == "all_files" and items and items[0][0] > 0:
        items = [item for item in items if item[0] > 0 or item[3] == "manifest"]
    return items[:max_candidates], parsed_genres, series_routes, series_evidence


def _parse_json_object(text):
    cleaned = _clean_llama_text(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _compact_text(text, max_chars=1200):
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= max_chars:
        return text
    half = max(100, (max_chars - 32) // 2)
    return f"{text[:half]} ... {text[-half:]}"


def _compact_list(values, max_items=6):
    return _as_string_list(values)[:max_items]


def _compact_sample(sample, max_chars=140):
    sample = re.sub(r"\s+", " ", str(sample or "")).strip()
    if len(sample) <= max_chars:
        return sample
    return sample[: max_chars - 3].rstrip() + "..."


def _selector_allowed_item(score, token, entry, source, reasons):
    item = {
        "token": token,
        "score": score,
    }

    if source == "manifest":
        matched = []
        matched.extend(reasons.get("strong_outfit_matches", []))
        matched.extend(reasons.get("strong_genre_matches", []))
        matched.extend(reasons.get("weak_genre_matches", []))
        matched.extend(reasons.get("series_strong_matches", []))
        matched.extend(reasons.get("series_weak_matches", []))
        matched.extend(reasons.get("series_name_matches", []))
        matched.extend(reasons.get("outfit_matches", []))
        matched.extend(reasons.get("prompt_clue_matches", []))
        matched.extend(reasons.get("category_matches", []))
        matched.extend(reasons.get("water_priority_matches", []))
        if matched:
            item["matches"] = _compact_list(matched, 8)
        else:
            manifest = entry.get("manifest") or {}
            tags = _compact_list(manifest.get("strong_tags"), 4)
            tags.extend(_compact_list(manifest.get("weak_tags"), 3))
            if tags:
                item["tags"] = _compact_list(tags, 7)
    else:
        item["name"] = entry.get("name", "")
        samples = [
            _compact_sample(sample)
            for sample in entry.get("samples", [])[:1]
            if str(sample).strip()
        ]
        if samples:
            item["sample"] = samples[0]
    return item


def _build_selector_user_msg(category_hint, parsed_genres, series_routes, max_selections, prompt, allowed):
    prompt_for_llm = _compact_text(prompt, 1200)
    compact_allowed = list(allowed)
    policy = [
        "Use the score and matches as routing evidence.",
        "Hard outfit matches are tier-1 routing evidence and should beat weak tone or demographic genres.",
        "Series routes are high-confidence filtered metadata; use them below explicit prompt and hard outfit evidence.",
        "Prefer high-scoring manifest candidates unless the prompt contradicts them.",
        "Select 1 to 3 complementary archetype files when several fit.",
        "Do not choose school or academy only because the subject is anime, manga, a girl, or young-looking.",
        "Choose school/academy only with explicit school, classroom, student, teacher, uniform, campus, academy, class, or club evidence.",
        "Choose magic academy only with both magic/fantasy and school/academy evidence.",
        "Prefer concrete prompt cues over genre stereotypes.",
    ]

    def dump(current_prompt, current_allowed):
        return json.dumps({
            "task": "Select context-appropriate wildcard tokens.",
            "category_hint": category_hint,
            "genres": parsed_genres[:20],
            "series_routes": series_routes[:24],
            "max_selections": max_selections,
            "prompt": current_prompt,
            "selection_policy": policy,
            "allowed_tokens": current_allowed,
            "required_json_schema": {
                "selected": ["__WildcardName__"],
                "reason": "short reason",
            },
        }, ensure_ascii=False, separators=(",", ":"))

    user_msg = dump(prompt_for_llm, compact_allowed)
    while len(user_msg) > 6000 and len(compact_allowed) > 8:
        compact_allowed = compact_allowed[: max(8, int(len(compact_allowed) * 0.75))]
        user_msg = dump(prompt_for_llm, compact_allowed)
    if len(user_msg) > 6000:
        prompt_for_llm = _compact_text(prompt, 700)
        user_msg = dump(prompt_for_llm, compact_allowed)
    return user_msg, compact_allowed


def _deterministic_selected_tokens(candidate_debug, max_selections, valid_tokens):
    ranked = [
        item for item in candidate_debug
        if item.get("token") in valid_tokens
    ]
    if not ranked:
        return []

    selected = []
    top_score = ranked[0].get("score", 0)
    for index, item in enumerate(ranked):
        score = item.get("score", 0)
        if index == 0:
            selected.append(item["token"])
        elif score >= 60 and (top_score <= 0 or score >= top_score * 0.5):
            selected.append(item["token"])
        if len(selected) >= max_selections:
            break
    return selected


class WildcardCatalog:
    DESCRIPTION = (
        "Scans wildcard .txt files and builds a catalog for the selector/resolver. "
        "Manifest and routing patch JSON files add metadata for smarter location selection."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "wildcard_roots": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Optional. One wildcard folder per line. Leave empty for common folders.",
                    "tooltip": (
                        "Optional wildcard folder paths, one per line. Leave empty to scan common ComfyUI "
                        "wildcard folders. If a folder contains manifest.json or *_routing_patch.json, "
                        "that metadata is used for smarter routing."
                    ),
                }),
                "max_files": ("INT", {
                    "default": 500,
                    "min": 1,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "Maximum number of wildcard .txt files to scan across all roots.",
                }),
                "sample_lines": ("INT", {
                    "default": 12,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": (
                        "How many example lines to keep per wildcard file for LLM selection. "
                        "The resolver still reads the full file when expanding a token."
                    ),
                }),
                "sample_mode": (["first", "random", "evenly_spaced"], {
                    "default": "first",
                    "tooltip": (
                        "How examples are chosen: first lines, deterministic pseudo-random lines, "
                        "or evenly spaced lines across the file."
                    ),
                }),
                "sample_seed": ("INT", {
                    "default": 0,
                    "step": 1,
                    "tooltip": "Seed used only when sample_mode is random. Same seed gives the same catalog examples.",
                }),
            }
        }

    RETURN_TYPES = ("WILDCARD_CATALOG", "STRING")
    RETURN_NAMES = ("catalog", "summary")
    FUNCTION = "load_catalog"
    CATEGORY = "VLM Nodes/Wildcards"

    def load_catalog(self, wildcard_roots, max_files, sample_lines, sample_mode="first", sample_seed=0):
        catalog = build_catalog(
            wildcard_roots,
            _as_int(max_files, 500, 1),
            _as_int(sample_lines, 12, 0),
            sample_mode,
            _as_int(sample_seed, 0),
        )
        summary = {
            "roots": catalog["roots"],
            "manifests": catalog.get("manifests", []),
            "routing_patches": catalog.get("routing_patches", []),
            "token_count": len(catalog["tokens"]),
            "tokens": sorted(catalog["tokens"].keys()),
        }
        return (catalog, json.dumps(summary, indent=2))


class WildcardResolver:
    DESCRIPTION = (
        "Expands wildcard tokens such as __LocationsFantasy__ and inline choices such as {a|b|c}. "
        "Use this when you want the VLM node to resolve the random entry itself."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "catalog": ("WILDCARD_CATALOG", {
                    "tooltip": "Catalog output from Wildcard Catalog."
                }),
                "text": ("STRING", {
                    "forceInput": True,
                    "default": "",
                    "tooltip": "Text containing wildcard tokens and/or dynamic prompt choices to expand.",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "step": 1,
                    "tooltip": "Seed for deterministic random wildcard line and {a|b|c} choice selection.",
                }),
                "missing_mode": (["keep", "remove", "error"], {
                    "default": "keep",
                    "tooltip": (
                        "What to do when a wildcard token is not found: keep the token, remove it, "
                        "or raise an error."
                    ),
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("resolved_text", "debug_json")
    FUNCTION = "resolve"
    CATEGORY = "VLM Nodes/Wildcards"

    def resolve(self, catalog, text, seed, missing_mode):
        resolved, used = resolve_text(catalog, text, seed, missing_mode)
        return (resolved, json.dumps({"used": used}, indent=2))


class LLMWildcardSelector:
    DESCRIPTION = (
        "Asks the connected LLM to choose context-appropriate wildcard tokens from a catalog. "
        "Manifest metadata, routing patches, genres, and filtered series tags are pre-scored before the LLM sees candidates."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "forceInput": True,
                    "default": "",
                    "tooltip": "Image prompt or character prompt used to infer the best wildcard location/category.",
                }),
                "catalog": ("WILDCARD_CATALOG", {
                    "tooltip": "Catalog output from Wildcard Catalog. This constrains selection to real wildcard files."
                }),
                "model": ("CUSTOM", {
                    "forceInput": True,
                    "default": "",
                    "tooltip": "Loaded llama.cpp text model to use for final wildcard selection.",
                }),
                "category_hint": ("STRING", {
                    "default": "location",
                    "tooltip": "Short hint for what kind of wildcard to select, usually location.",
                }),
                "max_candidates": ("INT", {
                    "default": 80,
                    "min": 1,
                    "max": 500,
                    "step": 1,
                    "tooltip": (
                        "Maximum scored candidates before compacting for the LLM. With manifests, the actual LLM "
                        "candidate list is capped further to keep context small."
                    ),
                }),
                "max_selections": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Maximum wildcard tokens to return. Use 1 for a single location, 2-3 for blends.",
                }),
                "fallback_token": ("STRING", {
                    "default": "",
                    "tooltip": "Optional wildcard token/name to use if the LLM and deterministic scoring produce no valid selection.",
                }),
                "output_mode": (["token_only", "resolved", "both"], {
                    "default": "token_only",
                    "tooltip": (
                        "token_only returns __Wildcard__. resolved expands to a random line. both returns token plus resolved text."
                    ),
                }),
                "seed": ("INT", {
                    "default": 0,
                    "step": 1,
                    "tooltip": "Seed for LLM sampling and deterministic wildcard resolving when output_mode resolves text.",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "LLM selection randomness. Keep at 0 for stable routing.",
                }),
                "manifest_mode": (["auto", "manifest_only", "all_files"], {
                    "default": "auto",
                    "tooltip": (
                        "auto prefers manifest/routing-patch files when present. manifest_only ignores unlisted files. "
                        "all_files lets every scanned wildcard compete."
                    ),
                }),
            },
            "optional": {
                "genres": ("STRING", {
                    "forceInput": True,
                    "default": "",
                    "tooltip": "Optional database genre labels, comma/newline/JSON list accepted. Medium-weight routing evidence.",
                }),
                "series_tags": ("STRING", {
                    "forceInput": True,
                    "default": "",
                    "tooltip": (
                        "Optional filtered routing tags from your tag node or LLM prefilter. High-weight metadata such as "
                        "sci-fi, futuristic city, school life, musical band, urban fantasy, high school."
                    ),
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("selected_text", "resolved_text", "debug_json")
    FUNCTION = "select"
    CATEGORY = "VLM Nodes/Wildcards"

    def select(self, prompt, catalog, model, category_hint, max_candidates, max_selections,
               fallback_token, output_mode, seed, temperature, manifest_mode="auto", genres="", series_tags=""):
        candidates, parsed_genres, series_routes, series_evidence = _candidate_catalog(
            catalog,
            prompt,
            category_hint,
            _as_int(max_candidates, 80, 1, 500),
            genres,
            manifest_mode,
            series_tags,
        )
        allowed = []
        candidate_debug = []
        candidate_cap = 24 if catalog.get("manifests") and manifest_mode != "all_files" else 36
        for score, token, entry, source, reasons in candidates[:candidate_cap]:
            allowed.append(_selector_allowed_item(score, token, entry, source, reasons))
            candidate_debug.append({
                "token": token,
                "score": score,
                "source": source,
                "reasons": reasons,
            })

        max_selections = _as_int(max_selections, 1, 1, 20)
        system_msg = (
            "You select wildcard tokens for image prompts. "
            "Return strict JSON only. Select only from allowed token strings. "
            "Do not invent wildcard names. Do not include reasoning, thought text, markdown, or commentary."
        )
        user_msg, allowed = _build_selector_user_msg(
            category_hint,
            parsed_genres,
            series_routes,
            max_selections,
            prompt,
            allowed,
        )

        response = _create_llama_text_response(
            model,
            [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            f"{system_msg}\n\n{user_msg}",
            max_tokens=256,
            temperature=_as_float(temperature, 0.0, 0.0, 1.0),
            top_p=1.0,
            top_k=40,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            repeat_penalty=1.05,
            seed=seed,
            sampling_mode=True,
            min_p=0.0,
            thinking=False,
            use_default_template=True,
        )
        text = _llama_text_response(response)
        parsed = _parse_json_object(text)
        selected = parsed.get("selected", [])
        if isinstance(selected, str):
            selected = [selected]

        valid_tokens = set(catalog.get("tokens", {}).keys())
        selected = [_token(token) for token in selected]
        selected = [token for token in selected if token in valid_tokens]
        selected = selected[:max_selections]
        selection_source = "llm_json" if selected else "none"

        if not selected and fallback_token:
            fallback = _token(fallback_token)
            if fallback in valid_tokens:
                selected = [fallback]
                selection_source = "fallback_token"

        if not selected:
            selected = _deterministic_selected_tokens(candidate_debug, max_selections, valid_tokens)
            if selected:
                selection_source = "deterministic_score"

        wildcard_text = ", ".join(selected)
        resolved_text = ""
        used = []
        if selected and output_mode in ("resolved", "both"):
            resolved_text, used = resolve_text(catalog, wildcard_text, seed, "error")
        if output_mode == "resolved":
            wildcard_output = resolved_text
        elif output_mode == "both" and resolved_text:
            wildcard_output = f"{wildcard_text}, {resolved_text}"
        else:
            wildcard_output = wildcard_text

        debug = {
            "raw_response": _clean_llama_text(text),
            "parsed": parsed,
            "selected": selected,
            "selection_source": selection_source,
            "resolved": resolved_text,
            "used": used,
            "candidate_count": len(allowed),
            "candidate_scores": candidate_debug,
            "selector_prompt_chars": len(user_msg),
            "genres": parsed_genres,
            "series_routes": series_routes,
            "series_tag_evidence": series_evidence[:80],
            "roots": catalog.get("roots", []),
            "manifests": catalog.get("manifests", []),
            "routing_patches": catalog.get("routing_patches", []),
        }
        return (wildcard_output, resolved_text, json.dumps(debug, indent=2, ensure_ascii=False))


NODE_CLASS_MAPPINGS = {
    "WildcardCatalog": WildcardCatalog,
    "WildcardResolver": WildcardResolver,
    "LLMWildcardSelector": LLMWildcardSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WildcardCatalog": "Wildcard Catalog",
    "WildcardResolver": "Wildcard Resolver",
    "LLMWildcardSelector": "LLM Wildcard Selector",
}
