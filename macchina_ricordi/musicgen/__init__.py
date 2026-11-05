"""The MusicGen engine: AudioCraft/Dora plumbing the nouns share.

Project config and recipes (``config``), runtime setup and checks (``setup``, ``runtime``, ``cli``), the generation
backend and inference patches (``backend``, ``fastgen``, ``sampling``), the shared AudioCraft dataset config
(``templates``, ``xp_datasource``) and small helpers (``shared``). What uses the engine lives with its noun:
``macchina_ricordi.data``, ``training``, ``models``, ``instrument``, ``tidy``.
"""
