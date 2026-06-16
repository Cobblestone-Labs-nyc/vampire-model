"""Exact vampire generation prompt + negative, lifted verbatim from the live config
(handoff §3). Single 'cathedral aristocratic vampire' look — the only scope.

Keep these in sync with /var/www/christopher/server.js if the live prompt ever changes.
"""

# Default prompt the live ip-adapter-face-id teacher uses.
VAMPIRE_PROMPT = (
    "portrait of the same person as an elegant aristocratic vampire, pale luminous skin, "
    "subtle sharp fangs, dark slicked-back hair, ornate gothic Victorian vampire attire "
    "with a high collar and dark velvet, standing inside a candlelit gothic cathedral with "
    "stained glass windows and tall stone arches, dramatic cinematic chiaroscuro lighting, "
    "photorealistic, highly detailed"
)

# Negative prompt — guards the mouth-melt / wrong-person / gender-shift failure modes.
NEGATIVE_PROMPT = (
    "cartoon, anime, illustration, deformed, distorted face, melted mouth, malformed teeth, "
    "extra limbs, extra fingers, mutated hands, different person, changed gender, "
    "low quality, blurry, watermark, text, signature"
)
