-- Brings an existing database up to the current schema.
--
-- `schema.sql` only runs when Postgres initialises an empty data directory, so
-- a stack that is already running needs this applied by hand:
--
--   docker compose exec -T postgres psql -U reserchia -d reserchia < docker/migrate.sql
--
-- Every statement is IF NOT EXISTS, so it is safe to run repeatedly and safe to
-- run against a database created from the current schema.sql.

-- Chainlit 2.11.1 writes these; the published schema predates them. Missing
-- columns do not raise -- the data layer logs a warning and the step is simply
-- never stored, which shows up as chat history not being saved.
ALTER TABLE steps ADD COLUMN IF NOT EXISTS "autoCollapse" BOOLEAN;
ALTER TABLE steps ADD COLUMN IF NOT EXISTS "icon" TEXT;

ALTER TABLE elements ADD COLUMN IF NOT EXISTS "autoPlay" BOOLEAN;
ALTER TABLE elements ADD COLUMN IF NOT EXISTS "playerConfig" JSONB;
ALTER TABLE elements ADD COLUMN IF NOT EXISTS "path" TEXT;
