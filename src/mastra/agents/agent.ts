import { pathToFileURL } from 'node:url';
import { Agent } from '@mastra/core/agent';
import { TaskSignalProvider } from '@mastra/core/signals';
import { askUserTool, createTool, webFetchTool } from '@mastra/core/tools';
import {
  createWorkspaceTools,
  LocalFilesystem,
  LocalSandbox,
  WORKSPACE_TOOLS,
  Workspace,
} from '@mastra/core/workspace';
import { Memory } from '@mastra/memory';
import { startScheduleTool, stopScheduleTool } from '../tools/schedule-tools';
import { ollama } from 'ollama-ai-provider-v2';
import { z } from 'zod';

const workspacePath = 'workspace';

const workspace = new Workspace({
  id: 'agent-workspace',
  name: 'Agent Workspace',
  filesystem: new LocalFilesystem({
    basePath: workspacePath,
  }),
  sandbox: new LocalSandbox({
    workingDirectory: workspacePath,
  }),
  tools: {
    [WORKSPACE_TOOLS.FILESYSTEM.WRITE_FILE]: {
      requireReadBeforeWrite: true,
    },
    [WORKSPACE_TOOLS.FILESYSTEM.EDIT_FILE]: {
      requireReadBeforeWrite: true,
    },
    [WORKSPACE_TOOLS.FILESYSTEM.DELETE]: {
      requireApproval: true,
    },
  },
});

const workspaceTools = await createWorkspaceTools(workspace);
const goalVerificationTools = {
  [WORKSPACE_TOOLS.FILESYSTEM.READ_FILE]: workspaceTools[WORKSPACE_TOOLS.FILESYSTEM.READ_FILE],
  [WORKSPACE_TOOLS.FILESYSTEM.LIST_FILES]: workspaceTools[WORKSPACE_TOOLS.FILESYSTEM.LIST_FILES],
  [WORKSPACE_TOOLS.FILESYSTEM.FILE_STAT]: workspaceTools[WORKSPACE_TOOLS.FILESYSTEM.FILE_STAT],
  [WORKSPACE_TOOLS.FILESYSTEM.GREP]: workspaceTools[WORKSPACE_TOOLS.FILESYSTEM.GREP],
};

// ollama-ai-provider-v2 resolves its own copy of @ai-sdk/provider, so its type is
// structurally compatible but nominally distinct from Mastra's expected model type
const localModel = ollama('gemma4:26b') as any;

const setRailsGoalTool = createTool({
  id: 'set_rails_goal',
  description:
    'Start a durable goal for the current memory thread before implementing a Rails application task. Preserve the user\'s requested scope in the objective.',
  inputSchema: z.object({
    objective: z.string().min(1).describe('The user\'s concrete Rails application task.'),
  }),
  execute: async ({ objective }, context) => {
    const { threadId, resourceId } = context.agent ?? {};

    if (!threadId) {
      return 'Cannot start the goal because this run has no memory-backed thread.';
    }

    const taskAgent = context.mastra?.getAgent('agent');
    if (!taskAgent) {
      return 'Cannot start the goal because the agent is not registered with Mastra.';
    }

    const record = await taskAgent.setObjective(objective, { threadId, resourceId });
    return record
      ? `Goal started for thread ${threadId}. Continue working until the goal is verified complete.`
      : 'Could not persist the goal. Confirm that Mastra storage and thread memory are configured.';
  },
});

export const agent = new Agent({
  id: 'agent',
  name: 'Ollama - gemma4:26b',
  description:
    'A general-purpose assistant that can research, manage tasks, work with local files, run approved commands, and create recurring schedules.',
  metadata: {
    suggestedPrompts: [
      "What's the weather in Austin this weekend?",
      "What's the SPCX stock price right now?",
      'Build a Japanese sakura festival landing page.',
    ], 
  },
  instructions: `You are a friendly starter agent for exploring what Mastra can do. Help the user try useful capabilities, build small projects, answer current questions, and shape this harness into a starting point for future work.

Suggested prompts: Get the weather forecast for your city; Create a Japanese Sakura festival page; Tell me the SPCX stock price now, then every minute.

When the user greets you or does not have a specific task, invite them to try the suggested prompts.

Ask concise questions when something is unclear or a good question could surface a useful insight.

For a Rails application coding task, first call set_rails_goal with the user's requested outcome and constraints. For a new Rails project, use RSpec rather than Minitest and configure SimpleCov to enforce 100% coverage for first-party application code. Implement the task in the workspace, add or update RSpec examples, run the full RSpec suite, and inspect the coverage report. Do not claim completion until the suite passes and coverage is 100% under the project's configured coverage metrics. Never weaken coverage thresholds, remove relevant specs, or add broad exclusions just to reach 100%. If execution is blocked, report the concrete blocker and leave the goal incomplete.

For local file changes, end with a plain-text URL using ${pathToFileURL(`${workspacePath}/`).href}; avoid Markdown links, localhost, /workspace, relative paths, and static-file servers.
`,
  // model: 'openai/gpt-5.6-terra',
  model: localModel,
  goal: {
    judge: localModel,
    maxRuns: 30,
    maxSteps: 5,
    tools: goalVerificationTools,
    prompt:
      'For Rails application tasks, use the read-only workspace tools to inspect the project implementation, RSpec configuration, and latest coverage report. Mark the goal complete only when the requested behavior is implemented, the full RSpec suite has actually run and passed, and the coverage report demonstrates 100% coverage. Treat missing test output, stale reports, claimed-but-unverified results, reduced thresholds, and broad coverage exclusions as incomplete. If the task is blocked by missing dependencies or unavailable infrastructure, do not mark it complete.',
  },
  defaultOptions: {
    maxSteps: 100,
    autoResumeSuspendedTools: true,
  },
  memory: new Memory({
    options: {
      generateTitle: true,
      observationalMemory: {
        model: 'openai/gpt-5-mini',
      },
    },
  }),
  workspace,
  tools: {
    ask_user: askUserTool,
    set_rails_goal: setRailsGoalTool,
    start_schedule: startScheduleTool,
    stop_schedule: stopScheduleTool,
    web_fetch: webFetchTool,
  },
  signals: [new TaskSignalProvider()],
});

export { setRailsGoalTool };
