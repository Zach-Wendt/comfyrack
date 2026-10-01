"""comfyrack public API.

    from comfyrack import Rack
    rack = Rack()
    rack.run("character-scene", recipe="anima_v1", overrides={"seed": 7})

Orchestrators needing structural graph surgery use load() for the manifest,
runner.load_graph(manifest, client) for the Graph, runner.patch() for mutations,
then runner.submit() instead of run()."""
from . import discover as _discover
from . import errors, out, preflight, provenance, recipes, runner  # noqa: F401
from .config import Config
from .http import ComfyClient
from .lexicon import Lexicon
from .profile import Profile
from .registry import Registry

__version__ = "0.1.0"


class Rack:
    def __init__(self, start_dir=None, machine: str | None = None,
                 builtin_dir=None, user_dir=None, user_config=None):
        self.config = Config.load(start_dir=start_dir, user_config=user_config)
        self.registry = Registry(self.config, builtin_dir=builtin_dir, user_dir=user_dir)
        self._machine = machine
        self._clients: dict[str, ComfyClient] = {}

    def client(self, machine: str | None = None) -> ComfyClient:
        url = self.config.machine_url(machine or self._machine)
        if url not in self._clients:
            self._clients[url] = ComfyClient(url)
        return self._clients[url]

    def client_for_url(self, url: str) -> ComfyClient:
        if url not in self._clients:
            self._clients[url] = ComfyClient(url)
        return self._clients[url]

    # -- lookup ------------------------------------------------------------

    def list(self, family: str | None = None):
        return self.registry.discover(family=family)

    def find(self, query: str):
        return self.registry.find(query)

    def describe(self, name: str):
        return self.registry.load(name)

    load = describe

    # -- execution ---------------------------------------------------------

    def run(self, name: str, recipe=None, overrides=None, machine=None, out=None,
            skip_preflight: bool = False, url: str | None = None,
            on_progress=None):
        """`machine` names a `[machines]` config entry and is resolved via
        `Config.machine_url(machine)`. `url` bypasses that resolution entirely
        and targets the URL directly via `client_for_url()` -- for a caller
        (batch.run_batch) that already resolved a URL itself (e.g. through
        `Config.machine_url(None)`'s env/default-fallback branch, which
        `machine_url(name)` does NOT retrace for an arbitrary name) and must
        not re-resolve it through the other, name-keyed branch. `url` wins if
        both are given.

        `on_progress` takes one preformatted line per progress event and is
        silent by default. A CLI passes one that writes to STDERR -- stdout
        carries the machine-readable result and must stay parseable.

        This is also the only layer that can resolve `[paths] loras`, which
        preflight's lineage check needs; an unconfigured path leaves that check
        dormant rather than failing the run."""
        recipe_values, recipe_name = None, None
        if isinstance(recipe, str):
            recipe_values, recipe_name = recipes.load_recipe(self.config, recipe), recipe
        elif isinstance(recipe, dict):
            recipe_values = recipe
        elif recipe is not None:
            raise errors.UsageError(
                f"recipe must be a recipe name (str) or a values dict, not {type(recipe).__name__}",
                help_text="A recipe must be either a string (recipe name to load) or a dict of parameter values."
            )
        results_dir = self.config.path("results")
        client = self.client_for_url(url) if url is not None else self.client(machine)
        return runner.run(self.registry, client, name,
                          recipe=recipe_values, overrides=overrides, out_path=out,
                          results_dir=results_dir, skip_preflight=skip_preflight,
                          recipe_name=recipe_name,
                          loras_dir=self.config.path("loras"),
                          on_progress=on_progress)

    def preflight(self, name: str, machine=None):
        client = self.client(machine)
        manifest = self.registry.load(name)
        graph = runner.load_graph(manifest, client)
        return runner.preflight_graph(self.registry, client, manifest, graph,
                                      loras_dir=self.config.path("loras"))

    # -- discovery ---------------------------------------------------------

    def nodes(self, pattern=None, machine=None):
        return _discover.node_types(self.client(machine), pattern)

    def node(self, class_type: str, machine=None):
        return _discover.node_info(self.client(machine), class_type)

    def models(self, type_=None, machine=None):
        """Model categories from /object_info, plus an "embeddings" category.

        Embeddings live behind their own route (see discover.embeddings), so
        they are merged in HERE rather than in the CLI: `models --type embeddings`
        and the bare `models` listing must agree on one source of truth, and a
        caller reading Rack.models() programmatically gets the same categories
        the CLI renders. An empty embedding list adds no category at all -- a
        machine with none installed must look the same as before."""
        client = self.client(machine)
        if type_ == "embeddings":
            return {"embeddings": _discover.embeddings(client)}
        grouped = _discover.models(client, type_)
        if type_ is None:
            names = _discover.embeddings(client)
            if names:
                grouped["embeddings"] = names
        return grouped

    def queue(self, machine=None):
        return _discover.queue_summary(self.client(machine))

    def stats(self, machine=None):
        return _discover.stats_summary(self.client(machine))

    @property
    def lexicon(self) -> Lexicon:
        if getattr(self, "_lexicon", None) is None:
            self._lexicon = Lexicon(self.config)
        return self._lexicon

    @property
    def profile(self) -> Profile:
        if getattr(self, "_profile", None) is None:
            self._profile = Profile(self.config)
        return self._profile

    def assemble_prompt(self, lineage: str, dialect: str, subject=None, location=None,
                        scene=None, action: str = "", style: str = "", allow_unknown: bool = False,
                        seed: int | None = None):
        """Returns (prompt_text, warnings). Wildcards expand here so the result is
        deterministic and recordable."""
        import random

        from . import guard, prompt as _prompt

        loc = scene or location
        rng = random.Random(seed)
        action = _prompt.expand_wildcards(action, rng=rng,
                                          wildcards_dir=self.config.path("wildcards"))
        findings = guard.check_terms(action, self.lexicon)
        if subject and not self.lexicon.has(subject):
            findings.append(guard.Finding("error", subject, "not in lexicon"))
        if loc and not self.lexicon.has(loc):
            findings.append(guard.Finding("error", loc, "not in lexicon"))
        warnings = guard.enforce(findings, allow_unknown=allow_unknown)

        if allow_unknown:
            text = _prompt.assemble(
                self.lexicon, lineage, dialect,
                subject=subject if subject and self.lexicon.has(subject) else None,
                location=loc if loc and self.lexicon.has(loc) else None,
                action=action, style=style)
            if subject and not self.lexicon.has(subject):
                text = f"{subject}, {text}" if text else subject
            if loc and not self.lexicon.has(loc) and loc != subject:
                text = f"{text}, {loc}" if text else loc
        else:
            text = _prompt.assemble(self.lexicon, lineage, dialect, subject=subject,
                                    location=loc, action=action, style=style)
        return text, warnings
