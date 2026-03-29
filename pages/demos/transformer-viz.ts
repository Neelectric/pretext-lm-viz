/*
Transformer Forward Pass Visualization - Llama-2 Style Decoder

Architecture (bottom to top):
- Embedding
- N × Transformer Layers:
  - RMS Norm → Self-Attention → + (residual)
  - RMS Norm → Feed Forward → + (residual)
- Final RMS Norm
- Linear (LM Head)
- Softmax

Tokens stream in using HuggingFace tokenizer, text wraps around the architecture.
*/

import {
  layoutNextLine,
  prepareWithSegments,
  type LayoutCursor,
  type PreparedTextWithSegments,
} from '../../src/layout.js'
import {
  carveTextLineSlots,
  getRectIntervalsForBand,
  type Interval,
  type Rect,
} from './wrap-geometry.js'
import { AutoTokenizer } from '@huggingface/transformers'

const NUM_LAYERS = 4

const BODY_FONT = '16px "Inter", "SF Pro Text", -apple-system, BlinkMacSystemFont, sans-serif'
const BODY_LINE_HEIGHT = 26

const SAMPLE_TEXT = `The transformer processes input tokens through a series of layers. Each layer contains two main components: a multi-head self-attention mechanism that allows tokens to attend to each other, and a feed-forward neural network (MLP) that processes each position independently. The attention mechanism computes queries, keys, and values from the input, then uses scaled dot-product attention to weight the values. Layer normalization and residual connections help with training stability. After all layers, the final hidden states are projected to vocabulary logits for next-token prediction.`

// Component types for the forward pass animation
type ComponentId =
  | 'embed'
  | `layer-${number}-norm1`
  | `layer-${number}-attn`
  | `layer-${number}-add1`
  | `layer-${number}-norm2`
  | `layer-${number}-ffn`
  | `layer-${number}-add2`
  | 'final-norm'
  | 'linear'
  | 'softmax'

type BandObstacle = {
  kind: 'rects'
  rects: Rect[]
  horizontalPadding: number
  verticalPadding: number
}

type PositionedLine = {
  x: number
  y: number
  width: number
  text: string
}

type ForwardPassState = {
  activeComponent: ComponentId | null
  stepIndex: number
  stepStartTime: number  // When the current step started
  phase: 'idle' | 'forward' | 'complete'
}

type TokenStreamState = {
  tokens: string[]
  visibleCount: number
  phase: 'idle' | 'streaming' | 'complete'
}

// Build the sequence of components to highlight during forward pass
function buildForwardPassSequence(): ComponentId[] {
  const sequence: ComponentId[] = ['embed']

  for (let i = 0; i < NUM_LAYERS; i++) {
    sequence.push(
      `layer-${i}-norm1`,
      `layer-${i}-attn`,
      `layer-${i}-add1`,
      `layer-${i}-norm2`,
      `layer-${i}-ffn`,
      `layer-${i}-add2`,
    )
  }

  sequence.push('final-norm', 'linear', 'softmax')
  return sequence
}

const FORWARD_SEQUENCE = buildForwardPassSequence()

// Gradual speed-up: each forward pass gets faster
// Forward pass 1 = 100ms per component, pass 2 = 50ms, etc.
const PASS_DURATIONS = [100, 50, 25, 15, 10, 8, 5, 3, 2, 1] // ms per component for each pass
function getStepDuration(forwardPassIndex: number): number {
  if (forwardPassIndex < PASS_DURATIONS.length) {
    return PASS_DURATIONS[forwardPassIndex]!
  }
  return PASS_DURATIONS[PASS_DURATIONS.length - 1]! // 5ms for all subsequent passes
}

const stageNode = document.getElementById('stage')
if (!(stageNode instanceof HTMLDivElement)) throw new Error('#stage not found')
const stage = stageNode

const btnForward = document.getElementById('btn-forward')
const btnReset = document.getElementById('btn-reset')

const animationState: ForwardPassState = {
  activeComponent: null,
  stepIndex: 0,
  stepStartTime: 0,
  phase: 'idle',
}

const tokenState: TokenStreamState = {
  tokens: [],
  visibleCount: 0,
  phase: 'idle',
}

const domCache: {
  architecture: HTMLDivElement | null
  componentElements: Map<ComponentId, HTMLElement>
  layerConnectors: Map<number, HTMLElement>  // Maps layer index to connector above it
  bodyLines: HTMLDivElement[]
  architectureBounds: Rect | null
} = {
  architecture: null,
  componentElements: new Map(),
  layerConnectors: new Map(),
  bodyLines: [],
  architectureBounds: null,
}

const preparedByKey = new Map<string, PreparedTextWithSegments>()
const scheduled = { value: false }

// Load tokenizer
let tokenizer: Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>> | null = null

async function loadTokenizer() {
  try {
    tokenizer = await AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-135M-Instruct')
    console.log('Tokenizer loaded successfully')
  } catch (e) {
    console.error('Failed to load tokenizer:', e)
    // Fallback: split by spaces
    tokenizer = null
  }
}

function tokenizeText(text: string): string[] {
  if (tokenizer) {
    const encoded = tokenizer.encode(text)
    // Decode each token individually to get the string representation
    const tokens: string[] = []
    for (let i = 0; i < encoded.length; i++) {
      const tokenId = encoded[i]
      if (tokenId === undefined) continue
      const decoded = tokenizer.decode([tokenId as number], { skip_special_tokens: true })
      if (decoded) tokens.push(decoded)
    }
    return tokens
  }
  // Fallback: split by word boundaries but keep spaces attached
  return text.match(/\S+\s*/g) || []
}

function getPrepared(text: string, font: string): PreparedTextWithSegments {
  const key = `${font}::${text}`
  const cached = preparedByKey.get(key)
  if (cached !== undefined) return cached
  const prepared = prepareWithSegments(text, font)
  preparedByKey.set(key, prepared)
  return prepared
}

function createComponent(
  id: ComponentId,
  label: string,
  type: 'embed' | 'norm' | 'attention' | 'ffn' | 'linear' | 'softmax',
): HTMLDivElement {
  const div = document.createElement('div')
  div.className = `component component--${type}`
  div.textContent = label
  div.dataset['componentId'] = id
  domCache.componentElements.set(id, div)
  return div
}

function createResidualAdd(id: ComponentId): HTMLDivElement {
  const div = document.createElement('div')
  div.className = 'residual-add'
  div.textContent = '+'
  div.dataset['componentId'] = id
  domCache.componentElements.set(id, div)
  return div
}

// Create an L-shaped SVG connector between staggered layers
// direction: 'left-to-right' | 'right-to-left' | 'center-to-right' | 'left-to-center'
function createStaggeredConnector(
  direction: 'left-to-right' | 'right-to-left' | 'center-to-right' | 'left-to-center',
  forLayerIndex: number,
): HTMLDivElement {
  const container = document.createElement('div')
  container.className = 'staggered-connector'
  container.dataset['forLayer'] = String(forLayerIndex)

  // The connector goes: up from source, horizontal to align, up to target
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
  svg.setAttribute('class', 'connector-svg')
  svg.setAttribute('viewBox', '0 0 60 20')
  svg.setAttribute('preserveAspectRatio', 'none')

  const path = document.createElementNS('http://www.w3.org/2000/svg', 'path')
  path.setAttribute('class', 'connector-path')

  switch (direction) {
    case 'right-to-left':
      // From right-side layer to left-side layer (going up)
      // Start at right (60), go up, left, up
      path.setAttribute('d', 'M 60 20 L 60 12 L 0 12 L 0 0')
      break
    case 'left-to-right':
      // From left-side layer to right-side layer (going up)
      // Start at left (0), go up, right, up
      path.setAttribute('d', 'M 0 20 L 0 12 L 60 12 L 60 0')
      break
    case 'center-to-right':
      // From center (embed) to right (L0)
      // Start at center (30), go up, right, up
      path.setAttribute('d', 'M 30 20 L 30 12 L 60 12 L 60 0')
      break
    case 'left-to-center':
      // From left (L3) to center (final-norm)
      // Start at left (0), go up, to center, up
      path.setAttribute('d', 'M 0 20 L 0 12 L 30 12 L 30 0')
      break
  }

  svg.appendChild(path)
  container.appendChild(svg)

  return container
}

// v2 - simplified, no connectors for debugging
function createTransformerLayer(layerIndex: number): HTMLDivElement {
  const block = document.createElement('div')
  block.className = 'layer-block'
  block.style.gap = '2px' // Use gap instead of connectors

  const label = document.createElement('span')
  label.className = 'layer-label'
  label.textContent = `L${layerIndex}`
  block.appendChild(label)

  // Data flow BOTTOM-TO-TOP: norm1 → attn → add1 → norm2 → ffn → add2
  // DOM appends TOP-TO-BOTTOM (first = top, last = bottom)
  // So append order: add2, ffn, norm2, add1, attn, norm1

  block.appendChild(createResidualAdd(`layer-${layerIndex}-add2`))   // TOP
  block.appendChild(createComponent(`layer-${layerIndex}-ffn`, 'Feed Forward', 'ffn'))
  block.appendChild(createComponent(`layer-${layerIndex}-norm2`, 'RMS Norm', 'norm'))
  block.appendChild(createResidualAdd(`layer-${layerIndex}-add1`))
  block.appendChild(createComponent(`layer-${layerIndex}-attn`, 'Self-Attention', 'attention'))
  block.appendChild(createComponent(`layer-${layerIndex}-norm1`, 'RMS Norm', 'norm'))  // BOTTOM

  return block
}

function createArchitectureDOM(): HTMLDivElement {
  const container = document.createElement('div')
  container.className = 'architecture'
  container.style.gap = '2px' // Use gap instead of connectors

  domCache.componentElements.clear()
  domCache.layerConnectors.clear()

  // Top: Output label and softmax
  const outputLabel = document.createElement('div')
  outputLabel.className = 'io-label'
  outputLabel.textContent = '↑ Output Probabilities'
  container.appendChild(outputLabel)

  container.appendChild(createComponent('softmax', 'Softmax', 'softmax'))
  container.appendChild(createComponent('linear', 'Linear', 'linear'))
  container.appendChild(createComponent('final-norm', 'RMS Norm', 'norm'))

  // Connector between final-norm and top layer (L3, which is on the left)
  // Goes from left (L3) to center (final-norm)
  const topConnector = createStaggeredConnector('left-to-center', NUM_LAYERS)
  container.appendChild(topConnector)
  domCache.layerConnectors.set(NUM_LAYERS, topConnector)

  // Layers container with Nx label
  const layersContainer = document.createElement('div')
  layersContainer.className = 'layers-container'
  layersContainer.style.gap = '0' // We'll add connectors manually

  const nxLabel = document.createElement('span')
  nxLabel.className = 'nx-label'
  nxLabel.textContent = `${NUM_LAYERS}×`
  layersContainer.appendChild(nxLabel)

  // Add layers with staggered positioning
  // L0, L2 on right; L1, L3 on left
  // Processing order: L0 → L1 → L2 → L3 (bottom to top)
  // Visual order (top to bottom): L3, L2, L1, L0
  for (let i = NUM_LAYERS - 1; i >= 0; i--) {
    // Add staggered connector above each layer (except the topmost)
    if (i < NUM_LAYERS - 1) {
      // Connector between layer i (below) and layer i+1 (above)
      // Even layers (0, 2) are on right, odd layers (1, 3) are on left
      const direction = i % 2 === 0 ? 'right-to-left' : 'left-to-right'
      const connector = createStaggeredConnector(direction, i)
      layersContainer.appendChild(connector)
      domCache.layerConnectors.set(i, connector)
    }

    const layer = createTransformerLayer(i)
    // Stagger: even layers (0, 2) on right, odd layers (1, 3) on left
    const offset = i % 2 === 0 ? 20 : -20
    layer.style.transform = `translateX(${offset}px)`
    layersContainer.appendChild(layer)
  }

  container.appendChild(layersContainer)

  // Connector between embedding and L0 (which is on the right)
  // Lights up when layer 0's norm1 lights up (entry to first layer)
  const embedConnector = createStaggeredConnector('center-to-right', -1)
  container.appendChild(embedConnector)
  domCache.layerConnectors.set(-1, embedConnector)

  // Bottom: Embedding
  container.appendChild(createComponent('embed', 'Embedding', 'embed'))

  const inputLabel = document.createElement('div')
  inputLabel.className = 'io-label'
  inputLabel.textContent = '↑ Input Tokens'
  container.appendChild(inputLabel)

  return container
}

function getObstacleIntervals(obstacle: BandObstacle, bandTop: number, bandBottom: number): Interval[] {
  return getRectIntervalsForBand(
    obstacle.rects,
    bandTop,
    bandBottom,
    obstacle.horizontalPadding,
    obstacle.verticalPadding,
  )
}

function layoutTextAroundObstacle(
  prepared: PreparedTextWithSegments,
  region: Rect,
  lineHeight: number,
  obstacles: BandObstacle[],
): PositionedLine[] {
  let cursor: LayoutCursor = { segmentIndex: 0, graphemeIndex: 0 }
  let lineTop = region.y
  const lines: PositionedLine[] = []

  while (cursor.segmentIndex < prepared.segments.length) {
    if (lineTop + lineHeight > region.y + region.height) break

    const bandTop = lineTop
    const bandBottom = lineTop + lineHeight
    const blocked: Interval[] = []

    for (const obstacle of obstacles) {
      const intervals = getObstacleIntervals(obstacle, bandTop, bandBottom)
      for (const interval of intervals) {
        blocked.push(interval)
      }
    }

    const slots = carveTextLineSlots(
      { left: region.x, right: region.x + region.width },
      blocked,
    )

    if (slots.length === 0) {
      lineTop += lineHeight
      continue
    }

    // Sort slots left to right
    slots.sort((a, b) => a.left - b.left)

    // Try to fill each slot on this line from left to right
    let filledAnySlot = false
    for (const slot of slots) {
      if (cursor.segmentIndex >= prepared.segments.length) break

      const width = slot.right - slot.left
      if (width < 40) continue // Skip very narrow slots

      const line = layoutNextLine(prepared, cursor, width)
      if (line === null) continue

      lines.push({
        x: Math.round(slot.left),
        y: Math.round(lineTop),
        width: line.width,
        text: line.text,
      })

      cursor = line.end
      filledAnySlot = true
    }

    // Move to next line
    if (!filledAnySlot) {
      // No slots could be filled, skip this line
    }
    lineTop += lineHeight
  }

  return lines
}

function syncPool<T extends HTMLElement>(
  pool: T[],
  length: number,
  create: () => T,
  parent: HTMLElement = stage,
): void {
  while (pool.length < length) {
    const element = create()
    pool.push(element)
    parent.appendChild(element)
  }
  while (pool.length > length) {
    const element = pool.pop()!
    element.remove()
  }
}

function projectBodyLines(
  lines: PositionedLine[],
  font: string,
  lineHeight: number,
): void {
  syncPool(domCache.bodyLines, lines.length, () => {
    const element = document.createElement('div')
    element.className = 'line'
    return element
  })

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!
    const element = domCache.bodyLines[i]!
    element.textContent = line.text
    element.style.left = `${line.x}px`
    element.style.top = `${line.y}px`
    element.style.font = font
    element.style.lineHeight = `${lineHeight}px`
  }
}

function updateComponentHighlights(): void {
  for (const [id, element] of domCache.componentElements) {
    const isActive = animationState.activeComponent === id
    element.classList.toggle('active', isActive)
  }

  // Update layer connectors
  // Connector at index N is between layer N (below) and layer N+1 (above)
  // It should light up when ENTERING layer N+1, i.e., when layer (N+1)'s norm1 is active
  for (const [layerIndex, connector] of domCache.layerConnectors) {
    let isActive = false
    if (layerIndex === -1) {
      // Embed connector: between embed and L0, fires when entering L0
      isActive = animationState.activeComponent === 'layer-0-norm1'
    } else if (layerIndex === NUM_LAYERS) {
      // Top connector: between top layer and final-norm, fires when entering final-norm
      isActive = animationState.activeComponent === 'final-norm'
    } else {
      // Inter-layer connector at index N: between layer N and layer N+1
      // Fires when entering layer N+1, i.e., when layer (N+1)'s norm1 is active
      isActive = animationState.activeComponent === `layer-${layerIndex + 1}-norm1`
    }
    connector.classList.toggle('active', isActive)
  }
}

function getVisibleText(): string {
  if (tokenState.phase === 'idle' || tokenState.tokens.length === 0) {
    return ''
  }
  return tokenState.tokens.slice(0, tokenState.visibleCount).join('')
}

function commitFrame(): boolean {
  const root = document.documentElement
  const pageWidth = root.clientWidth
  const pageHeight = root.clientHeight
  const gutter = 48
  const centerX = pageWidth / 2
  const centerY = pageHeight / 2

  // Create architecture DOM if needed
  if (domCache.architecture === null) {
    domCache.architecture = createArchitectureDOM()
    stage.appendChild(domCache.architecture)
  }

  // Position architecture in center
  domCache.architecture.style.left = `${centerX}px`
  domCache.architecture.style.top = `${centerY}px`

  updateComponentHighlights()

  // Get architecture bounding rect for obstacle routing
  const archRect = domCache.architecture.getBoundingClientRect()
  const stageRect = stage.getBoundingClientRect()

  const obstacleRect: Rect = {
    x: archRect.left - stageRect.left,
    y: archRect.top - stageRect.top,
    width: archRect.width,
    height: archRect.height,
  }

  domCache.architectureBounds = obstacleRect

  const obstacles: BandObstacle[] = [
    {
      kind: 'rects',
      rects: [obstacleRect],
      horizontalPadding: 30,
      verticalPadding: 10,
    },
  ]

  const textRegion: Rect = {
    x: gutter,
    y: gutter,
    width: pageWidth - gutter * 2,
    height: pageHeight - gutter * 2 - 60, // Leave room for controls
  }

  // Get the visible text based on token streaming state
  const visibleText = getVisibleText()

  if (visibleText) {
    const preparedBody = getPrepared(visibleText, BODY_FONT)
    const lines = layoutTextAroundObstacle(
      preparedBody,
      textRegion,
      BODY_LINE_HEIGHT,
      obstacles,
    )
    projectBodyLines(lines, BODY_FONT, BODY_LINE_HEIGHT)
  } else {
    // Clear lines if no text
    syncPool(domCache.bodyLines, 0, () => document.createElement('div'))
  }

  const isAnimating = animationState.phase === 'forward' || tokenState.phase === 'streaming'
  return isAnimating
}

function render(now: number): boolean {
  // Update forward pass animation with dynamic step durations
  // Speed is based on which forward pass we're on (how many tokens generated)
  if (animationState.phase === 'forward') {
    const forwardPassIndex = tokenState.visibleCount // 0 for first pass, 1 for second, etc.
    const stepDuration = getStepDuration(forwardPassIndex)
    const elapsed = now - animationState.stepStartTime

    // Check if current step is complete
    if (elapsed >= stepDuration) {
      // Advance to next step
      animationState.stepIndex++
      animationState.stepStartTime = now

      if (animationState.stepIndex >= FORWARD_SEQUENCE.length) {
        // Forward pass complete - spawn a new token!
        animationState.phase = 'complete'
        animationState.activeComponent = null
        animationState.stepIndex = 0

        // Spawn the next token when forward pass completes
        if (tokenState.phase === 'streaming' && tokenState.visibleCount < tokenState.tokens.length) {
          tokenState.visibleCount++

          // If there are more tokens, start another forward pass
          if (tokenState.visibleCount < tokenState.tokens.length) {
            animationState.phase = 'forward'
            animationState.stepIndex = 0
            animationState.activeComponent = FORWARD_SEQUENCE[0]!
            animationState.stepStartTime = now
          } else {
            tokenState.phase = 'complete'
          }
        }
      } else {
        animationState.activeComponent = FORWARD_SEQUENCE[animationState.stepIndex]!
      }
    }
  }

  return commitFrame()
}

function scheduleRender(): void {
  if (scheduled.value) return
  scheduled.value = true
  requestAnimationFrame(function renderFrame(now) {
    scheduled.value = false
    if (render(now)) scheduleRender()
  })
}

function startForwardPass(): void {
  // Start fresh if idle or complete
  if (tokenState.phase === 'idle' || tokenState.phase === 'complete') {
    tokenState.tokens = tokenizeText(SAMPLE_TEXT)
    tokenState.visibleCount = 0
    tokenState.phase = 'streaming'

    // Start the first forward pass - token spawns when it completes
    animationState.phase = 'forward'
    animationState.stepIndex = 0
    animationState.activeComponent = FORWARD_SEQUENCE[0]!
    animationState.stepStartTime = performance.now()
  }

  scheduleRender()
}

function resetAnimation(): void {
  animationState.phase = 'idle'
  animationState.activeComponent = null
  animationState.stepIndex = 0

  tokenState.phase = 'idle'
  tokenState.visibleCount = 0
  tokenState.tokens = []

  // Clear prepared text cache
  preparedByKey.clear()

  // Clear displayed lines
  syncPool(domCache.bodyLines, 0, () => document.createElement('div'))

  scheduleRender()
}

window.addEventListener('resize', scheduleRender)
btnForward?.addEventListener('click', startForwardPass)
btnReset?.addEventListener('click', resetAnimation)

// Initialize
await document.fonts.ready
await loadTokenizer()
commitFrame()
