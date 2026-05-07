# Layered patches over the published Holocene DA image. Building locally
# in CI means changes to scripts/ ship without rebuilding the env each
# run — the FROM layer is cached on the runner; only the COPY rebuilds.
#
# Anything in scripts/ overrides the container's baked-in /app/* copy.

FROM davidedge/lipd_webapps:holocene_da

COPY scripts/ /app/
