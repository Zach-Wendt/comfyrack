from comfyrack import dispatch, errors


def cap(name, loras=(), checkpoints=(), classes=("KSampler", "LoraLoader")):
    return dispatch.Capability(
        machine=name, url=f"http://{name}:8188",
        models={"loras": list(loras), "checkpoints": list(checkpoints)},
        node_classes=set(classes))


CAPS = {
    "local": cap("local", loras=["char_a.safetensors", "char_b.safetensors"],
                 checkpoints=["anima_baseV10.safetensors"]),
    "remote": cap("remote", loras=["char_a.safetensors"],
                  checkpoints=["anima_baseV10.safetensors"]),
}


def test_index_builds_a_capability_per_machine():
    class FakeClient:
        def __init__(self, url):
            self.url = url

        def object_info(self, class_type=None):
            return {"KSampler": {"input": {"required": {}}},
                    "CheckpointLoaderSimple": {"input": {"required": {
                        "ckpt_name": [["anima_baseV10.safetensors"], {}]}}}}

    caps, errs = dispatch.index({"local": "http://local:8188"}, lambda url: FakeClient(url))
    assert caps["local"].url == "http://local:8188"
    assert "KSampler" in caps["local"].node_classes
    assert caps["local"].models["checkpoints"] == ["anima_baseV10.safetensors"]
    assert errs == {}


def test_index_skips_an_unreachable_machine_and_reports_it_instead_of_aborting():
    """One offline box must not take down indexing of the machines that ARE up --
    the same rule run_queue follows for job failures -- and it must not vanish
    silently either: it has to show up in the returned errors."""
    class FakeClient:
        def __init__(self, url):
            self.url = url

        def object_info(self, class_type=None):
            return {"KSampler": {"input": {"required": {}}}}

    class DeadClient:
        def __init__(self, url):
            self.url = url

        def object_info(self, class_type=None):
            raise errors.ComfyrackError(f"cannot reach ComfyUI at {self.url}")

    def factory(url):
        return DeadClient(url) if "dead" in url else FakeClient(url)

    caps, errs = dispatch.index(
        {"local": "http://local:8188", "dead": "http://dead:8188"}, factory)
    assert "local" in caps
    assert "dead" not in caps
    assert isinstance(errs["dead"], errors.ComfyrackError)


def test_a_machine_holding_every_required_model_is_capable():
    assert dispatch.capable(CAPS["local"], {"loras": ["char_b.safetensors"]}, set())


def test_a_machine_missing_a_required_lora_is_not_capable():
    assert not dispatch.capable(CAPS["remote"], {"loras": ["char_b.safetensors"]}, set())


def test_a_machine_missing_a_required_node_class_is_not_capable():
    assert not dispatch.capable(CAPS["local"], {}, {"SomeCustomNode"})


def test_route_returns_every_capable_machine():
    assert set(dispatch.route({"loras": ["char_a.safetensors"]}, CAPS)) == \
        {"local", "remote"}


def test_route_excludes_machines_lacking_the_model():
    assert dispatch.route({"loras": ["char_b.safetensors"]}, CAPS) == ["local"]


def test_route_with_no_capable_machine_returns_empty():
    assert dispatch.route({"loras": ["nobody.safetensors"]}, CAPS) == []


def test_queue_runs_every_job_and_records_the_machine_used():
    jobs = [dispatch.Job("s01", {"loras": ["char_a.safetensors"]}, None),
            dispatch.Job("s02", {"loras": ["char_b.safetensors"]}, None)]
    calls = []

    def runner(job, machine):
        calls.append((job.shot_name, machine))
        return {"ok": True}

    results = dispatch.run_queue(jobs, CAPS, runner)
    assert {r["shot"] for r in results} == {"s01", "s02"}
    assert dict(calls)["s02"] == "local"      # only local has char_b
    assert all(r["status"] == "ok" for r in results)


def test_a_job_with_no_capable_machine_is_reported_not_raised():
    jobs = [dispatch.Job("s01", {"loras": ["nobody.safetensors"]}, None)]
    results = dispatch.run_queue(jobs, CAPS, lambda job, machine: {"ok": True})
    assert results[0]["status"] == "unroutable"
    assert "nobody.safetensors" in results[0]["detail"]


def test_unroutable_detail_names_only_files_missing_on_every_machine():
    """local holds char_b.safetensors; remote does not. Neither holds
    nobody.safetensors. The union of what any one machine lacks is
    {char_b, nobody} -- which would wrongly send the user chasing char_b, a file
    `local` already has. Only nobody.safetensors is missing EVERYWHERE, so
    only it belongs in the message (the documented remedy for `unroutable` is
    "copy the missing file to a machine that lacks it")."""
    jobs = [dispatch.Job(
        "s01", {"loras": ["char_b.safetensors", "nobody.safetensors"]}, None)]
    results = dispatch.run_queue(jobs, CAPS, lambda job, machine: {"ok": True})
    assert results[0]["status"] == "unroutable"
    assert "nobody.safetensors" in results[0]["detail"]
    assert "char_b.safetensors" not in results[0]["detail"]


def test_unroutable_detail_names_the_problem_when_no_machines_are_configured():
    jobs = [dispatch.Job("s01", {"loras": ["char_b.safetensors"]}, None)]
    results = dispatch.run_queue(jobs, {}, lambda job, machine: {"ok": True})
    assert results[0]["status"] == "unroutable"
    assert "no machines" in results[0]["detail"]


def test_a_failed_job_is_retried_on_another_capable_machine():
    attempted = []

    def runner(job, machine):
        attempted.append(machine)
        if len(attempted) == 1:
            raise RuntimeError("CUDA OOM")
        return {"ok": True}

    jobs = [dispatch.Job("s01", {"loras": ["char_a.safetensors"]}, None)]
    results = dispatch.run_queue(jobs, CAPS, runner, max_retries=1)
    assert results[0]["status"] == "ok"
    assert len(attempted) == 2
    assert attempted[0] != attempted[1]


def test_a_job_failing_everywhere_is_reported_as_failed_with_the_last_error():
    def runner(job, machine):
        raise RuntimeError("CUDA OOM")

    jobs = [dispatch.Job("s01", {"loras": ["char_a.safetensors"]}, None)]
    results = dispatch.run_queue(jobs, CAPS, runner, max_retries=1)
    assert results[0]["status"] == "failed"
    assert "OOM" in results[0]["detail"]


def test_a_failed_job_pairs_the_reported_machine_with_its_own_error():
    """local and remote are both capable of this job; rotation tries local then
    remote (sorted order), and both fail with distinguishable errors. The
    reported `machine` must be the one that actually produced the reported
    `detail` -- not the first machine tried paired with the last machine's
    error."""
    def runner(job, machine):
        raise RuntimeError(f"{machine} blew up")

    jobs = [dispatch.Job("s01", {"loras": ["char_a.safetensors"]}, None)]
    results = dispatch.run_queue(jobs, CAPS, runner, max_retries=1)
    assert results[0]["status"] == "failed"
    assert results[0]["detail"] == f"{results[0]['machine']} blew up"


def test_retry_with_only_one_capable_machine_retries_that_same_machine():
    """Only `local` holds char_b.safetensors. With one candidate, the retry
    rotation has nowhere else to go -- it should land on `local` again, not
    error out for lack of a second machine."""
    attempted = []

    def runner(job, machine):
        attempted.append(machine)
        if len(attempted) == 1:
            raise RuntimeError("transient")
        return {"ok": True}

    jobs = [dispatch.Job("s01", {"loras": ["char_b.safetensors"]}, None)]
    results = dispatch.run_queue(jobs, CAPS, runner, max_retries=1)
    assert results[0]["status"] == "ok"
    assert attempted == ["local", "local"]


def test_one_failure_does_not_stop_the_rest_of_the_queue():
    def runner(job, machine):
        if job.shot_name == "s01":
            raise RuntimeError("boom")
        return {"ok": True}

    jobs = [dispatch.Job("s01", {}, None), dispatch.Job("s02", {}, None)]
    results = dispatch.run_queue(jobs, CAPS, runner, max_retries=0)
    assert [r["status"] for r in results] == ["failed", "ok"]


# -- subfoldered model references (final review, Important 6) ----------------
#
# 13 of this project's 206 real LoRAs live in a subfolder (`Anima\
# AnimaEditV1.safetensors`). The two sides of a routing comparison come from
# different sources and do not have to agree about the subfolder -- or about the
# separator, since a Windows box and a Linux box report the same file
# differently -- and a raw `in` test made that disagreement fatal. Because
# "missing on every machine" is reported as `unroutable`, a job was declared
# PERMANENTLY unroutable over a spelling while the file sat on the box. The
# matching silent half of the same bug lived in runner.lora_lineages(); see
# test_carry_forward.py.

def test_a_subfoldered_lora_is_not_reported_missing_when_the_machine_holds_it():
    c = cap("local", loras=["Anima/AnimaEditV1.safetensors"])
    # A character recipe hand-writing the bare filename...
    assert dispatch.capable(c, {"loras": ["AnimaEditV1.safetensors"]}, set())
    # ...the full reference as the machine spells it...
    assert dispatch.capable(c, {"loras": ["Anima/AnimaEditV1.safetensors"]}, set())
    # ...and the same reference with the other host's separator.
    assert dispatch.capable(c, {"loras": ["Anima\\AnimaEditV1.safetensors"]}, set())


def test_a_requirement_carrying_a_subfolder_matches_a_machine_listing_it_bare():
    """The mirror case, so normalisation is not one-directional."""
    c = cap("local", loras=["AnimaEditV1.safetensors"])
    assert dispatch.capable(c, {"loras": ["Anima/AnimaEditV1.safetensors"]}, set())


def test_a_lora_the_machine_genuinely_lacks_is_still_reported_missing():
    """Normalising must not turn the check into a rubber stamp."""
    c = cap("local", loras=["Anima/AnimaEditV1.safetensors"])
    assert not dispatch.capable(c, {"loras": ["nobody.safetensors"]}, set())
    assert dispatch._missing(c, {"loras": ["nobody.safetensors"]},
                             set()) == ["nobody.safetensors"]


def test_a_bare_name_claimed_by_two_subfolders_matches_neither():
    """Same rule Manifest._build_alias_map follows for an ambiguous alias:
    silently picking one of two candidates resolves to the wrong file."""
    c = cap("local", loras=["Anima/x.safetensors", "Krea/x.safetensors"])
    assert not dispatch.capable(c, {"loras": ["x.safetensors"]}, set())
    # The unambiguous full reference still matches.
    assert dispatch.capable(c, {"loras": ["Anima/x.safetensors"]}, set())


def test_a_subfoldered_lora_present_somewhere_is_routable_rather_than_unroutable():
    """End of the chain: the spelling difference must not reach run_queue as an
    `unroutable` verdict, which the docs tell a user to fix by copying a file
    that is already there."""
    caps = {"local": cap("local", loras=["Anima/AnimaEditV1.safetensors"])}
    job = dispatch.Job(shot_name="s01",
                       requirements={"loras": ["AnimaEditV1.safetensors"]})
    results = dispatch.run_queue([job], caps, lambda j, m: {"paths": []})
    assert results[0]["status"] == "ok"
    assert dispatch.route(job.requirements, caps) == ["local"]
