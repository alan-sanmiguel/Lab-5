import { spawn } from 'node:child_process'

const children = []
let stopping = false

function stop(signal = 'SIGTERM') {
  if (stopping) return
  stopping = true
  for (const child of children) {
    if (child.exitCode === null) child.kill(signal)
  }
}

function start(command, args) {
  const child = spawn(command, args, { stdio: 'inherit', env: process.env })
  children.push(child)
  child.on('error', (error) => {
    console.error(`Could not start ${args.join(' ')}: ${error.message}`)
    stop()
  })
  child.on('exit', (code) => {
    if (!stopping) stop()
    if (code && code !== 0) process.exitCode = code
  })
  return child
}

async function waitForApi(server) {
  const deadline = Date.now() + 15_000
  while (Date.now() < deadline) {
    if (server.exitCode !== null) {
      throw new Error('The local API server exited before it was ready.')
    }
    try {
      const response = await fetch('http://127.0.0.1:3000/api/health', {
        signal: AbortSignal.timeout(500),
      })
      if (response.ok) return
    } catch {
      await new Promise((resolve) => setTimeout(resolve, 250))
    }
  }
  throw new Error('The local API server did not start within 15 seconds.')
}

process.on('SIGINT', () => stop('SIGINT'))
process.on('SIGTERM', () => stop('SIGTERM'))

try {
  const api = start(process.execPath, ['server/index.js'])
  await waitForApi(api)
  if (!stopping) {
    start(process.execPath, [
      'node_modules/vite/bin/vite.js',
      '--host',
      '127.0.0.1',
      '--strictPort',
    ])
  }
} catch (error) {
  console.error(error instanceof Error ? error.message : 'Could not start the app.')
  process.exitCode = 1
  stop()
}
