-- Chainlit's SQLAlchemy data layer schema.
--
-- Verbatim from https://docs.chainlit.io/data-layers/sqlalchemy for Chainlit
-- 2.11.1. Do not trim it: `command`, `defaultOpen` and `modes` on `steps` were
-- added in 2.1.0, 2.3.0 and 2.9.4 respectively, and an older copy of this file
-- starts cleanly and then fails mid-conversation when a step is written.
--
-- Postgres runs this once, on an empty data directory. To re-apply it after an
-- edit the volume must be dropped: `docker compose down -v`.

CREATE TABLE IF NOT EXISTS users (
    "id" UUID PRIMARY KEY,
    "identifier" TEXT NOT NULL UNIQUE,
    "metadata" JSONB NOT NULL,
    "createdAt" TEXT
);

CREATE TABLE IF NOT EXISTS threads (
    "id" UUID PRIMARY KEY,
    "createdAt" TEXT,
    "name" TEXT,
    "userId" UUID,
    "userIdentifier" TEXT,
    "tags" TEXT[],
    "metadata" JSONB,
    FOREIGN KEY ("userId") REFERENCES users("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS steps (
    "id" UUID PRIMARY KEY,
    "name" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "threadId" UUID NOT NULL,
    "parentId" UUID,
    "streaming" BOOLEAN NOT NULL,
    "waitForAnswer" BOOLEAN,
    "isError" BOOLEAN,
    "metadata" JSONB,
    "tags" TEXT[],
    "input" TEXT,
    "output" TEXT,
    "createdAt" TEXT,
    "command" TEXT,
    "start" TEXT,
    "end" TEXT,
    "generation" JSONB,
    "showInput" TEXT,
    "language" TEXT,
    "indent" INT,
    "defaultOpen" BOOLEAN,
    "modes" JSONB,
    -- Not in the published schema, and required by 2.11.1. The data layer builds
    -- its INSERT from whatever keys the step carries -- unfiltered -- so a key
    -- with no column aborts the write. It fails as a logged warning, not an
    -- error the UI shows, so the symptom is chat history silently not saving.
    "autoCollapse" BOOLEAN,
    "icon" TEXT,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS elements (
    "id" UUID PRIMARY KEY,
    "threadId" UUID,
    "type" TEXT,
    "url" TEXT,
    "chainlitKey" TEXT,
    "name" TEXT NOT NULL,
    "display" TEXT,
    "objectKey" TEXT,
    "size" TEXT,
    "page" INT,
    "language" TEXT,
    "forId" UUID,
    "mime" TEXT,
    "props" JSONB,
    -- Same reason as the step columns above: audio/video elements and uploaded
    -- files carry these, and an absent column fails the whole insert.
    "autoPlay" BOOLEAN,
    "playerConfig" JSONB,
    "path" TEXT,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS feedbacks (
    "id" UUID PRIMARY KEY,
    "forId" UUID NOT NULL,
    "threadId" UUID NOT NULL,
    "value" INT NOT NULL,
    "comment" TEXT,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
);

-- Not in the upstream schema. The sidebar lists a user's threads newest-first on
-- every page load, which is a sequential scan without this.
CREATE INDEX IF NOT EXISTS threads_user_created_idx
    ON threads ("userIdentifier", "createdAt" DESC);
CREATE INDEX IF NOT EXISTS steps_thread_idx ON steps ("threadId");
CREATE INDEX IF NOT EXISTS elements_thread_idx ON elements ("threadId");
