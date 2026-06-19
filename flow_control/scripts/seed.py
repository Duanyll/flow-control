def run(config: dict) -> None:
    """Generate a seed checkpoint.

    Resolves the trainer from ``launch.type`` and calls its ``seed_checkpoint()``.
    The base trainer seeds a transformer adapter (the common case); trainers that
    train something else (e.g. the VAE trainer) override it. Plugin trainers are
    already registered by this point (cli loads ``imports`` before dispatching).
    """
    from flow_control.training.launch_config import trainer_registry

    launch_type = config.get("launch", {}).get("type", "")
    trainer_cls = trainer_registry.get(launch_type)
    if trainer_cls is None:
        raise ValueError(
            f"Unknown trainer type {launch_type!r}. Registered: "
            f"{sorted(trainer_registry.tags())}. If it is a plugin trainer, add "
            "its module to the config's `imports`."
        )
    trainer = trainer_cls(**config)
    trainer.seed_checkpoint()
