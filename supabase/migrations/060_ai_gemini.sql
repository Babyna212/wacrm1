-- ============================================================
-- 060_ai_gemini.sql — add 'gemini' as a supported AI provider
-- ============================================================

-- Extend the check constraint on ai_configs to allow 'gemini'.
ALTER TABLE ai_configs
  DROP CONSTRAINT IF EXISTS ai_configs_provider_check;

ALTER TABLE ai_configs
  ADD CONSTRAINT ai_configs_provider_check
  CHECK (provider IN ('openai', 'anthropic', 'openai_compatible', 'gemini'));

-- Extend the matching constraint on ai_usage_log so usage
-- records for Gemini-generated replies can be stored.
ALTER TABLE ai_usage_log
  DROP CONSTRAINT IF EXISTS ai_usage_log_provider_check;

ALTER TABLE ai_usage_log
  ADD CONSTRAINT ai_usage_log_provider_check
  CHECK (provider IN ('openai', 'anthropic', 'openai_compatible', 'gemini'));

