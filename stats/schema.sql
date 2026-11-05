-- wrangler d1 execute neyra-stats --remote --file=schema.sql
CREATE TABLE IF NOT EXISTS reports (
  month TEXT NOT NULL,
  body TEXT NOT NULL,
  n INTEGER NOT NULL,
  PRIMARY KEY (month, body)
);
