import { writeFileSync } from 'node:fs'
import { type ChildProcess } from 'node:child_process'
import {
  createBrowserSession,
  ensurePageServer,
  loadHashReport,
  type BrowserKind,
} from './browser-automation.ts'

type BenchmarkResult = {
  label: string
  ms: number
  desc: string
}

type CorpusBenchmarkResult = {
  id: string
  label: string
  font: string
  chars: number
  segments: number
  width: number
  lineCount: number
  prepareMs: number
  layoutMs: number
}

type BenchmarkReport = {
  status: 'ready' | 'error'
  requestId?: string
  results?: BenchmarkResult[]
  corpusResults?: CorpusBenchmarkResult[]
  message?: string
}

function parseStringFlag(name: string): string | null {
  const prefix = `--${name}=`
  const arg = process.argv.find(value => value.startsWith(prefix))
  return arg === undefined ? null : arg.slice(prefix.length)
}

function parseNumberFlag(name: string, fallback: number): number {
  const raw = parseStringFlag(name)
  if (raw === null) return fallback
  const parsed = Number.parseInt(raw, 10)
  if (!Number.isFinite(parsed)) {
    throw new Error(`Invalid value for --${name}: ${raw}`)
  }
  return parsed
}

function parseBrowser(value: string | null): BrowserKind {
  const browser = (value ?? process.env['BENCHMARK_CHECK_BROWSER'] ?? 'chrome').toLowerCase()
  if (browser !== 'chrome' && browser !== 'safari') {
    throw new Error(`Unsupported browser ${browser}; expected chrome or safari`)
  }
  return browser
}

function printReport(report: BenchmarkReport): void {
  if (report.status === 'error') {
    console.log(`error: ${report.message ?? 'unknown error'}`)
    return
  }

  console.log('Top-level batch benchmark:')
  for (const result of report.results ?? []) {
    console.log(`  ${result.label}: ${result.ms < 0.01 ? '<0.01' : result.ms.toFixed(2)}ms`)
  }

  if ((report.corpusResults ?? []).length > 0) {
    console.log('Long-form corpus stress:')
    for (const corpus of report.corpusResults!) {
      console.log(
        `  ${corpus.label}: prepare ${corpus.prepareMs.toFixed(2)}ms | layout ${corpus.layoutMs < 0.01 ? '<0.01' : corpus.layoutMs.toFixed(2)}ms | ${corpus.segments.toLocaleString()} segs | ${corpus.lineCount} lines @ ${corpus.width}px`,
      )
    }
  }
}

const browser = parseBrowser(parseStringFlag('browser'))
const port = parseNumberFlag('port', Number.parseInt(process.env['BENCHMARK_CHECK_PORT'] ?? '3210', 10))
const output = parseStringFlag('output')

let serverProcess: ChildProcess | null = null
const session = createBrowserSession(browser, { foreground: true })

try {
  const pageServer = await ensurePageServer(port, '/benchmark', process.cwd())
  serverProcess = pageServer.process
  const baseUrl = `${pageServer.baseUrl}/benchmark`
  const requestId = `${Date.now()}-${Math.random().toString(36).slice(2)}`
  const url =
    `${baseUrl}?report=1` +
    `&requestId=${encodeURIComponent(requestId)}`

  const report = await loadHashReport<BenchmarkReport>(session, url, requestId, browser)
  printReport(report)

  if (output !== null) {
    writeFileSync(output, JSON.stringify(report, null, 2))
    console.log(`wrote ${output}`)
  }

  if (report.status === 'error') {
    process.exitCode = 1
  }
} finally {
  session.close()
  serverProcess?.kill()
}
