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
Layers are draggable with real-time text reflow.
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
import { pipeline } from '@huggingface/transformers'

// DistilGPT2 has 6 transformer layers
const NUM_LAYERS = 6
const LAYERS_PER_COLUMN = 3  // 2 columns × 3 layers = 6

const BODY_FONT = '16px "Inter", "SF Pro Text", -apple-system, BlinkMacSystemFont, sans-serif'
const BODY_LINE_HEIGHT = 26

// Default input prompt
const DEFAULT_PROMPT = `The transformer architecture`

// Model settings
const MODEL_ID = 'Xenova/distilgpt2'
const MAX_NEW_TOKENS = 100

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
  stepStartTime: number
  phase: 'idle' | 'forward' | 'complete'
  // Timing synchronization
  forwardPassDuration: number  // Measured duration of actual model forward pass
  lastMeasuredDuration: number // Rolling average of forward pass times
}

type TokenStreamState = {
  inputText: string           // The input prompt (shown in blue)
  generatedTokens: string[]   // Tokens generated so far (shown in default color)
  tokenQueue: string[]        // Queue of tokens waiting to be displayed
  phase: 'idle' | 'streaming' | 'complete'
  generationComplete: boolean // True when model has finished generating
}

// Position state for draggable elements
type DraggablePosition = {
  x: number  // Offset from default position
  y: number
}

type DragState = {
  active: boolean
  targetType: 'layer' | 'embed' | 'head' | null
  targetIndex: number  // Layer index, or -1 for embed, -2 for head
  startMouseX: number
  startMouseY: number
  startPosX: number
  startPosY: number
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

// Calculate step duration based on measured forward pass time
// Distribute the forward pass duration across all animation steps
function getStepDuration(): number {
  const totalSteps = FORWARD_SEQUENCE.length
  // Use measured duration, with a minimum to keep animation visible
  const duration = Math.max(animationState.forwardPassDuration, 100)
  return duration / totalSteps
}

const stageNode = document.getElementById('stage')
if (!(stageNode instanceof HTMLDivElement)) throw new Error('#stage not found')
const stage = stageNode

const btnForward = document.getElementById('btn-forward')
const btnReset = document.getElementById('btn-reset')

// Create editable prompt element
const promptEditor = document.createElement('div')
promptEditor.className = 'prompt-editor'
promptEditor.contentEditable = 'true'
promptEditor.spellcheck = false
promptEditor.textContent = DEFAULT_PROMPT
stage.appendChild(promptEditor)

// Get the current prompt from editor
function getPrompt(): string {
  return promptEditor.textContent?.trim() || DEFAULT_PROMPT
}

// Set the prompt text
function setPrompt(text: string): void {
  promptEditor.textContent = text
}

// Enable/disable prompt editor
function setPromptEnabled(enabled: boolean): void {
  promptEditor.contentEditable = enabled ? 'true' : 'false'
  promptEditor.classList.toggle('disabled', !enabled)
}

// Show/hide prompt editor
function setPromptVisible(visible: boolean): void {
  promptEditor.style.display = visible ? 'block' : 'none'
}

// Position prompt editor at text start position
function positionPromptEditor(x: number, y: number): void {
  promptEditor.style.left = `${x}px`
  promptEditor.style.top = `${y}px`
}

const animationState: ForwardPassState = {
  activeComponent: null,
  stepIndex: 0,
  stepStartTime: 0,
  phase: 'idle',
  forwardPassDuration: 500,  // Initial estimate (will be updated)
  lastMeasuredDuration: 500,
}

const tokenState: TokenStreamState = {
  inputText: DEFAULT_PROMPT,
  generatedTokens: [],
  tokenQueue: [],
  phase: 'idle',
  generationComplete: false,
}

// Draggable positions for each element
// Layer positions are offsets from their default staggered positions
const layerPositions: DraggablePosition[] = Array.from({ length: NUM_LAYERS }, () => ({ x: 0, y: 0 }))
const embedPosition: DraggablePosition = { x: 0, y: 0 }
const headPosition: DraggablePosition = { x: 0, y: 0 }  // For final-norm, linear, softmax group

const dragState: DragState = {
  active: false,
  targetType: null,
  targetIndex: -1,
  startMouseX: 0,
  startMouseY: 0,
  startPosX: 0,
  startPosY: 0,
}

const domCache: {
  layerElements: HTMLDivElement[]
  embedElement: HTMLDivElement | null
  headElement: HTMLDivElement | null
  connectorSvg: SVGSVGElement | null
  connectorPaths: SVGPathElement[]
  componentElements: Map<ComponentId, HTMLElement>
  bodyLines: HTMLDivElement[]
} = {
  layerElements: [],
  embedElement: null,
  headElement: null,
  connectorSvg: null,
  connectorPaths: [],
  componentElements: new Map(),
  bodyLines: [],
}

const preparedByKey = new Map<string, PreparedTextWithSegments>()
const scheduled = { value: false }

// Text generation pipeline
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let generator: any = null
let modelLoadPromise: Promise<boolean> | null = null

async function loadModel(): Promise<boolean> {
  // If already loading, return the existing promise
  if (modelLoadPromise) return modelLoadPromise

  modelLoadPromise = (async () => {
    try {
      console.log('Loading text generation pipeline...')
      generator = await pipeline('text-generation', MODEL_ID, {
        progress_callback: (progress: { status: string; progress?: number; file?: string }) => {
          if (progress.progress !== undefined && progress.file) {
            console.log(`Loading ${progress.file}: ${Math.round(progress.progress)}%`)
          }
        },
      })
      console.log('Model loaded successfully')
      return true
    } catch (e) {
      console.error('Failed to load model:', e)
      generator = null
      modelLoadPromise = null  // Allow retry
      return false
    }
  })()

  return modelLoadPromise
}

function isModelLoaded(): boolean {
  return generator !== null
}

// Generate tokens one at a time, synchronized with layer animation
async function runGeneration() {
  if (!generator) {
    console.error('Model not loaded')
    return
  }

  // Disable input during generation
  setPromptEnabled(false)

  try {
    const inputPrompt = tokenState.inputText
    let currentText = inputPrompt
    let tokenCount = 0

    while (tokenCount < MAX_NEW_TOKENS && tokenState.phase === 'streaming') {
      // Wait for current animation to finish before starting next forward pass
      while (animationState.phase === 'forward' && tokenState.phase === 'streaming') {
        await new Promise(resolve => setTimeout(resolve, 10))
      }

      if (tokenState.phase !== 'streaming') break

      // Start forward pass animation BEFORE running the model
      animationState.phase = 'forward'
      animationState.stepIndex = 0
      animationState.activeComponent = FORWARD_SEQUENCE[0]!
      animationState.stepStartTime = performance.now()
      scheduleRender()

      // Time the actual model forward pass
      const forwardStartTime = performance.now()

      const output = await generator(currentText, {
        max_new_tokens: 1,
        do_sample: false,
        return_full_text: true,
      })

      const forwardEndTime = performance.now()
      const actualDuration = forwardEndTime - forwardStartTime

      // Update measured duration (rolling average)
      animationState.lastMeasuredDuration =
        animationState.lastMeasuredDuration * 0.7 + actualDuration * 0.3
      animationState.forwardPassDuration = animationState.lastMeasuredDuration

      if (output && output[0] && output[0].generated_text) {
        const newFullText = output[0].generated_text as string
        const newToken = newFullText.slice(currentText.length)

        if (newToken) {
          // Check for EOS
          if (newToken.includes('<|endoftext|>') || newToken.includes('</s>')) {
            console.log('EOS token generated')
            tokenState.tokenQueue.push(newToken.replace('<|endoftext|>', '').replace('</s>', ''))
            break
          }

          tokenState.tokenQueue.push(newToken)
          currentText = newFullText
          tokenCount++
        } else {
          console.log('No new token generated, stopping')
          break
        }
      } else {
        break
      }
    }

    console.log(`Generation complete, ${tokenCount} tokens generated`)
  } catch (e) {
    console.error('Generation error:', e)
  } finally {
    // Re-enable input when done
    setPromptEnabled(true)
  }

  tokenState.generationComplete = true
  animationState.phase = 'complete'
  animationState.activeComponent = null
  scheduleRender()
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

function createTransformerLayer(layerIndex: number): HTMLDivElement {
  const block = document.createElement('div')
  block.className = 'layer-block draggable'
  block.style.gap = '2px'
  block.dataset['layerIndex'] = String(layerIndex)

  const label = document.createElement('span')
  label.className = 'layer-label'
  label.textContent = `L${layerIndex}`
  block.appendChild(label)

  // Data flow BOTTOM-TO-TOP: norm1 → attn → add1 → norm2 → ffn → add2
  // DOM appends TOP-TO-BOTTOM (first = top, last = bottom)
  block.appendChild(createResidualAdd(`layer-${layerIndex}-add2`))
  block.appendChild(createComponent(`layer-${layerIndex}-ffn`, 'Feed Forward', 'ffn'))
  block.appendChild(createComponent(`layer-${layerIndex}-norm2`, 'RMS Norm', 'norm'))
  block.appendChild(createResidualAdd(`layer-${layerIndex}-add1`))
  block.appendChild(createComponent(`layer-${layerIndex}-attn`, 'Self-Attention', 'attention'))
  block.appendChild(createComponent(`layer-${layerIndex}-norm1`, 'RMS Norm', 'norm'))

  return block
}

function createEmbedBlock(): HTMLDivElement {
  const block = document.createElement('div')
  block.className = 'embed-block draggable'
  block.style.display = 'flex'
  block.style.flexDirection = 'column'
  block.style.alignItems = 'center'
  block.style.gap = '2px'

  block.appendChild(createComponent('embed', 'Embedding', 'embed'))

  const inputLabel = document.createElement('div')
  inputLabel.className = 'io-label'
  inputLabel.textContent = '↑ Input Tokens'
  block.appendChild(inputLabel)

  return block
}

function createHeadBlock(): HTMLDivElement {
  const block = document.createElement('div')
  block.className = 'head-block draggable'
  block.style.display = 'flex'
  block.style.flexDirection = 'column'
  block.style.alignItems = 'center'
  block.style.gap = '2px'

  const outputLabel = document.createElement('div')
  outputLabel.className = 'io-label'
  outputLabel.textContent = '↑ Output Probabilities'
  block.appendChild(outputLabel)

  block.appendChild(createComponent('softmax', 'Softmax', 'softmax'))
  block.appendChild(createComponent('linear', 'Linear', 'linear'))
  block.appendChild(createComponent('final-norm', 'RMS Norm', 'norm'))

  return block
}

function createConnectorSvg(): SVGSVGElement {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
  svg.setAttribute('class', 'connector-overlay')
  svg.style.cssText = `
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 1;
  `

  // Create paths for each connector:
  // - embed to L0
  // - L0 to L1, L1 to L2, L2 to L3
  // - L3 to head
  for (let i = 0; i <= NUM_LAYERS; i++) {
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path')
    path.setAttribute('class', 'connector-path')
    path.setAttribute('fill', 'none')
    path.setAttribute('stroke', 'var(--border)')
    path.setAttribute('stroke-width', '2')
    path.setAttribute('stroke-linecap', 'round')
    path.setAttribute('stroke-linejoin', 'round')
    path.style.transition = 'stroke 100ms ease, filter 100ms ease'
    svg.appendChild(path)
    domCache.connectorPaths.push(path)
  }

  return svg
}

// Calculate the default center position for each element
// 3 columns × 4 layers each: L0-L3 (left), L4-L7 (middle), L8-L11 (right)
function getDefaultLayerCenter(layerIndex: number, pageWidth: number, pageHeight: number): { x: number, y: number } {
  const centerX = pageWidth / 2
  const centerY = pageHeight / 2

  // Which column (0=left, 1=middle, 2=right)
  const column = Math.floor(layerIndex / LAYERS_PER_COLUMN)
  // Position within column (0=bottom, 3=top)
  const posInColumn = layerIndex % LAYERS_PER_COLUMN

  // Vertical spacing
  const layerHeight = 90
  const layerGap = 32

  // Total height of one column
  const columnHeight = LAYERS_PER_COLUMN * layerHeight + (LAYERS_PER_COLUMN - 1) * layerGap

  // Column top (centered vertically)
  const columnTop = centerY - columnHeight / 2

  // Y position: bottom layer (pos 0) at bottom, top layer (pos 3) at top
  const layerY = columnTop + (LAYERS_PER_COLUMN - 1 - posInColumn) * (layerHeight + layerGap) + layerHeight / 2

  // Horizontal spacing between columns
  const columnGap = 150
  const numColumns = Math.ceil(NUM_LAYERS / LAYERS_PER_COLUMN)  // 5 columns

  // X position: center the columns (for 5 columns: -2, -1, 0, 1, 2)
  const columnOffset = column - (numColumns - 1) / 2
  const columnX = centerX + columnOffset * columnGap

  return { x: columnX, y: layerY }
}

function getDefaultEmbedCenter(pageWidth: number, pageHeight: number): { x: number, y: number } {
  // Embed is in the bottom center, lower than before
  const centerX = pageWidth / 2
  const l0Center = getDefaultLayerCenter(0, pageWidth, pageHeight)
  return { x: centerX, y: l0Center.y + 130 }
}

function getDefaultHeadCenter(pageWidth: number, pageHeight: number): { x: number, y: number } {
  // Head is above the middle column (centered), higher up
  const centerX = pageWidth / 2
  const l11Center = getDefaultLayerCenter(NUM_LAYERS - 1, pageWidth, pageHeight)
  return { x: centerX, y: l11Center.y - 130 }
}

// Get the actual center position (default + drag offset)
function getLayerCenter(layerIndex: number, pageWidth: number, pageHeight: number): { x: number, y: number } {
  const def = getDefaultLayerCenter(layerIndex, pageWidth, pageHeight)
  const offset = layerPositions[layerIndex]!
  return { x: def.x + offset.x, y: def.y + offset.y }
}

function getEmbedCenter(pageWidth: number, pageHeight: number): { x: number, y: number } {
  const def = getDefaultEmbedCenter(pageWidth, pageHeight)
  return { x: def.x + embedPosition.x, y: def.y + embedPosition.y }
}

function getHeadCenter(pageWidth: number, pageHeight: number): { x: number, y: number } {
  const def = getDefaultHeadCenter(pageWidth, pageHeight)
  return { x: def.x + headPosition.x, y: def.y + headPosition.y }
}

// Update connector paths based on current positions
function updateConnectorPaths(pageWidth: number, pageHeight: number): void {
  if (!domCache.connectorSvg) return

  const embedCenter = getEmbedCenter(pageWidth, pageHeight)
  const headCenter = getHeadCenter(pageWidth, pageHeight)

  // Path 0: embed to L0 (embed is center, L0 is left column)
  const l0Center = getLayerCenter(0, pageWidth, pageHeight)
  const path0 = domCache.connectorPaths[0]!
  const embedToL0 = createEmbedToL0Path(
    embedCenter.x, embedCenter.y - 25,
    l0Center.x, l0Center.y + 45,
  )
  path0.setAttribute('d', embedToL0)

  // Paths 1 to NUM_LAYERS-1: layer to layer
  for (let i = 0; i < NUM_LAYERS - 1; i++) {
    const fromCenter = getLayerCenter(i, pageWidth, pageHeight)
    const toCenter = getLayerCenter(i + 1, pageWidth, pageHeight)
    const path = domCache.connectorPaths[i + 1]!

    // Check if this is a cross-column connection (L3→L4, L7→L8)
    const fromColumn = Math.floor(i / LAYERS_PER_COLUMN)
    const toColumn = Math.floor((i + 1) / LAYERS_PER_COLUMN)
    const isCrossColumn = fromColumn !== toColumn

    let pathD: string
    if (isCrossColumn) {
      // Cross-column: route through the gap between columns
      // Gap X is midway between the two columns
      const gapX = (fromCenter.x + toCenter.x) / 2
      pathD = createCrossColumnPath(
        fromCenter.x, fromCenter.y - 45,  // Top of source layer
        toCenter.x, toCenter.y + 45,      // Bottom of target layer
        gapX,
      )
    } else {
      // Same column: straight vertical line
      pathD = createStraightPath(
        fromCenter.x, fromCenter.y - 45,
        toCenter.x, toCenter.y + 45,
      )
    }
    path.setAttribute('d', pathD)
  }

  // Path NUM_LAYERS: L11 to head (L11 is right column, head is center)
  const lastLayerCenter = getLayerCenter(NUM_LAYERS - 1, pageWidth, pageHeight)
  const pathLast = domCache.connectorPaths[NUM_LAYERS]!
  const lastToHead = createL11ToHeadPath(
    lastLayerCenter.x, lastLayerCenter.y - 45,
    headCenter.x, headCenter.y + 40,
  )
  pathLast.setAttribute('d', lastToHead)
}

// Create a straight vertical path (for same-column connections)
function createStraightPath(x1: number, y1: number, x2: number, y2: number): string {
  if (Math.abs(x1 - x2) < 1) {
    // Perfectly vertical
    return `M ${x1} ${y1} L ${x2} ${y2}`
  }
  // If not vertical, use L-shape
  const midY = (y1 + y2) / 2
  return `M ${x1} ${y1} L ${x1} ${midY} L ${x2} ${midY} L ${x2} ${y2}`
}

// Create embed to L0 path: up from embed, left to L0's x, up to L0
function createEmbedToL0Path(x1: number, y1: number, x2: number, y2: number): string {
  // Go up just a tiny bit, then left, then up to L0
  const midY = y1 - 8  // Go up only 8px first
  return `M ${x1} ${y1} L ${x1} ${midY} L ${x2} ${midY} L ${x2} ${y2}`
}

// Create a cross-column path (top of one column to bottom of next)
// Route: up from source, right to midpoint between columns, down below target, right to target x, up to target
function createCrossColumnPath(
  x1: number, y1: number,  // Top of source layer
  x2: number, y2: number,  // Bottom of target layer
  gapX: number,            // X position of the gap between columns
): string {
  // Go up 20px above source
  const topY = y1 - 20
  // Go down 20px below target
  const bottomY = y2 + 20

  // Path: up from source, right to gap, down below target, right to target x, up to target
  return `M ${x1} ${y1} L ${x1} ${topY} L ${gapX} ${topY} L ${gapX} ${bottomY} L ${x2} ${bottomY} L ${x2} ${y2}`
}

// Create L11 to head path: up quite a bit from L11, then left to center, then up to head
function createL11ToHeadPath(x1: number, y1: number, x2: number, y2: number): string {
  // Go up significantly before turning left
  const turnY = y2 + 20  // Turn at 20px below the head
  return `M ${x1} ${y1} L ${x1} ${turnY} L ${x2} ${turnY} L ${x2} ${y2}`
}

function updateConnectorHighlights(): void {
  const activeComponent = animationState.activeComponent

  // Path 0: embed to L0, lights up when entering L0
  const path0Active = activeComponent === 'layer-0-norm1'
  domCache.connectorPaths[0]?.setAttribute('stroke', path0Active ? 'var(--accent)' : 'var(--border)')
  if (path0Active) {
    domCache.connectorPaths[0]?.setAttribute('filter', 'drop-shadow(0 0 4px var(--accent-glow))')
  } else {
    domCache.connectorPaths[0]?.removeAttribute('filter')
  }

  // Paths 1 to NUM_LAYERS-1: light up when entering the target layer
  for (let i = 0; i < NUM_LAYERS - 1; i++) {
    const pathActive = activeComponent === `layer-${i + 1}-norm1`
    const path = domCache.connectorPaths[i + 1]
    path?.setAttribute('stroke', pathActive ? 'var(--accent)' : 'var(--border)')
    if (pathActive) {
      path?.setAttribute('filter', 'drop-shadow(0 0 4px var(--accent-glow))')
    } else {
      path?.removeAttribute('filter')
    }
  }

  // Path NUM_LAYERS: lights up when entering final-norm
  const pathLastActive = activeComponent === 'final-norm'
  const pathLast = domCache.connectorPaths[NUM_LAYERS]
  pathLast?.setAttribute('stroke', pathLastActive ? 'var(--accent)' : 'var(--border)')
  if (pathLastActive) {
    pathLast?.setAttribute('filter', 'drop-shadow(0 0 4px var(--accent-glow))')
  } else {
    pathLast?.removeAttribute('filter')
  }
}

function initializeDOMElements(): void {
  domCache.componentElements.clear()
  domCache.layerElements = []
  domCache.connectorPaths = []

  // Create connector SVG first (so it's behind elements)
  domCache.connectorSvg = createConnectorSvg()
  stage.appendChild(domCache.connectorSvg)

  // Create embed block
  domCache.embedElement = createEmbedBlock()
  domCache.embedElement.style.position = 'absolute'
  stage.appendChild(domCache.embedElement)

  // Create layer blocks
  for (let i = 0; i < NUM_LAYERS; i++) {
    const layer = createTransformerLayer(i)
    layer.style.position = 'absolute'
    domCache.layerElements.push(layer)
    stage.appendChild(layer)
  }

  // Create head block
  domCache.headElement = createHeadBlock()
  domCache.headElement.style.position = 'absolute'
  stage.appendChild(domCache.headElement)

  // Set up drag handlers
  setupDragHandlers()
}

function setupDragHandlers(): void {
  // Layer drag handlers
  for (let i = 0; i < domCache.layerElements.length; i++) {
    const layer = domCache.layerElements[i]!
    layer.addEventListener('mousedown', (e) => startDrag(e, 'layer', i))
  }

  // Embed drag handler
  domCache.embedElement?.addEventListener('mousedown', (e) => startDrag(e, 'embed', -1))

  // Head drag handler
  domCache.headElement?.addEventListener('mousedown', (e) => startDrag(e, 'head', -2))

  // Global mouse handlers
  document.addEventListener('mousemove', handleDragMove)
  document.addEventListener('mouseup', handleDragEnd)
}

function startDrag(e: MouseEvent, targetType: 'layer' | 'embed' | 'head', targetIndex: number): void {
  e.preventDefault()

  dragState.active = true
  dragState.targetType = targetType
  dragState.targetIndex = targetIndex
  dragState.startMouseX = e.clientX
  dragState.startMouseY = e.clientY

  if (targetType === 'layer') {
    const pos = layerPositions[targetIndex]!
    dragState.startPosX = pos.x
    dragState.startPosY = pos.y
  } else if (targetType === 'embed') {
    dragState.startPosX = embedPosition.x
    dragState.startPosY = embedPosition.y
  } else {
    dragState.startPosX = headPosition.x
    dragState.startPosY = headPosition.y
  }

  // Add dragging class for cursor
  document.body.classList.add('dragging')
}

function handleDragMove(e: MouseEvent): void {
  if (!dragState.active) return

  const dx = e.clientX - dragState.startMouseX
  const dy = e.clientY - dragState.startMouseY

  if (dragState.targetType === 'layer') {
    const pos = layerPositions[dragState.targetIndex]!
    pos.x = dragState.startPosX + dx
    pos.y = dragState.startPosY + dy
  } else if (dragState.targetType === 'embed') {
    embedPosition.x = dragState.startPosX + dx
    embedPosition.y = dragState.startPosY + dy
  } else if (dragState.targetType === 'head') {
    headPosition.x = dragState.startPosX + dx
    headPosition.y = dragState.startPosY + dy
  }

  scheduleRender()
}

function handleDragEnd(): void {
  if (!dragState.active) return

  dragState.active = false
  dragState.targetType = null
  document.body.classList.remove('dragging')

  scheduleRender()
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

    slots.sort((a, b) => a.left - b.left)

    for (const slot of slots) {
      if (cursor.segmentIndex >= prepared.segments.length) break

      const width = slot.right - slot.left
      if (width < 40) continue

      const line = layoutNextLine(prepared, cursor, width)
      if (line === null) continue

      lines.push({
        x: Math.round(slot.left),
        y: Math.round(lineTop),
        width: line.width,
        text: line.text,
      })

      cursor = line.end
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

// Project lines with input text in blue, generated text in default color
function projectBodyLinesWithColors(
  lines: PositionedLine[],
  font: string,
  lineHeight: number,
  inputLength: number,
): void {
  syncPool(domCache.bodyLines, lines.length, () => {
    const element = document.createElement('div')
    element.className = 'line'
    return element
  })

  let charOffset = 0
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!
    const element = domCache.bodyLines[i]!

    element.style.left = `${line.x}px`
    element.style.top = `${line.y}px`
    element.style.font = font
    element.style.lineHeight = `${lineHeight}px`

    const lineStart = charOffset
    const lineEnd = charOffset + line.text.length

    if (lineEnd <= inputLength) {
      // Entire line is input (blue)
      element.innerHTML = `<span class="input-text">${escapeHtml(line.text)}</span>`
    } else if (lineStart >= inputLength) {
      // Entire line is generated (default)
      element.innerHTML = escapeHtml(line.text)
    } else {
      // Line spans the boundary
      const inputPart = line.text.slice(0, inputLength - lineStart)
      const generatedPart = line.text.slice(inputLength - lineStart)
      element.innerHTML = `<span class="input-text">${escapeHtml(inputPart)}</span>${escapeHtml(generatedPart)}`
    }

    charOffset += line.text.length
  }
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

function updateComponentHighlights(): void {
  for (const [id, element] of domCache.componentElements) {
    const isActive = animationState.activeComponent === id
    element.classList.toggle('active', isActive)
  }

  updateConnectorHighlights()
}

function getVisibleText(): { input: string, generated: string } {
  return {
    input: tokenState.inputText,
    generated: tokenState.generatedTokens.join(''),
  }
}

// Get bounding rect for an element relative to stage
function getElementRect(element: HTMLElement): Rect {
  const rect = element.getBoundingClientRect()
  const stageRect = stage.getBoundingClientRect()
  return {
    x: rect.left - stageRect.left,
    y: rect.top - stageRect.top,
    width: rect.width,
    height: rect.height,
  }
}

function commitFrame(): boolean {
  const root = document.documentElement
  const pageWidth = root.clientWidth
  const pageHeight = root.clientHeight
  const gutter = 48

  // Initialize DOM if needed
  if (domCache.layerElements.length === 0) {
    initializeDOMElements()
  }

  // Position all elements based on current state
  // Layers
  for (let i = 0; i < NUM_LAYERS; i++) {
    const center = getLayerCenter(i, pageWidth, pageHeight)
    const element = domCache.layerElements[i]!
    element.style.left = `${center.x}px`
    element.style.top = `${center.y}px`
    element.style.transform = 'translate(-50%, -50%)'
  }

  // Embed
  if (domCache.embedElement) {
    const center = getEmbedCenter(pageWidth, pageHeight)
    domCache.embedElement.style.left = `${center.x}px`
    domCache.embedElement.style.top = `${center.y}px`
    domCache.embedElement.style.transform = 'translate(-50%, -50%)'
  }

  // Head
  if (domCache.headElement) {
    const center = getHeadCenter(pageWidth, pageHeight)
    domCache.headElement.style.left = `${center.x}px`
    domCache.headElement.style.top = `${center.y}px`
    domCache.headElement.style.transform = 'translate(-50%, -50%)'
  }

  // Update connector paths
  updateConnectorPaths(pageWidth, pageHeight)

  // Update highlights
  updateComponentHighlights()

  // Build obstacles from all draggable elements
  const obstacles: BandObstacle[] = []

  // Layer obstacles
  for (let i = 0; i < domCache.layerElements.length; i++) {
    const rect = getElementRect(domCache.layerElements[i]!)
    obstacles.push({
      kind: 'rects',
      rects: [rect],
      horizontalPadding: 20,
      verticalPadding: 8,
    })
  }

  // Embed obstacle
  if (domCache.embedElement) {
    const rect = getElementRect(domCache.embedElement)
    obstacles.push({
      kind: 'rects',
      rects: [rect],
      horizontalPadding: 20,
      verticalPadding: 8,
    })
  }

  // Head obstacle
  if (domCache.headElement) {
    const rect = getElementRect(domCache.headElement)
    obstacles.push({
      kind: 'rects',
      rects: [rect],
      horizontalPadding: 20,
      verticalPadding: 8,
    })
  }

  const textRegion: Rect = {
    x: gutter,
    y: gutter,
    width: pageWidth - gutter * 2,
    height: pageHeight - gutter * 2 - 60,
  }

  // Position prompt editor at text start
  positionPromptEditor(textRegion.x, textRegion.y)

  // When idle, show editable prompt; when generating, show text lines
  const isIdle = tokenState.phase === 'idle'

  if (isIdle) {
    // Show prompt editor, hide text lines
    setPromptVisible(true)
    syncPool(domCache.bodyLines, 0, () => document.createElement('div'))
  } else {
    // Hide prompt editor, show text lines
    setPromptVisible(false)

    const { input, generated } = getVisibleText()
    const fullText = input + generated

    if (fullText) {
      const preparedBody = getPrepared(fullText, BODY_FONT)
      const lines = layoutTextAroundObstacle(
        preparedBody,
        textRegion,
        BODY_LINE_HEIGHT,
        obstacles,
      )
      // Project lines with color split at input boundary
      projectBodyLinesWithColors(lines, BODY_FONT, BODY_LINE_HEIGHT, input.length)
    } else {
      syncPool(domCache.bodyLines, 0, () => document.createElement('div'))
    }
  }

  return shouldContinueAnimation()
}

function render(now: number): boolean {
  if (animationState.phase === 'forward') {
    const stepDuration = getStepDuration()
    const elapsed = now - animationState.stepStartTime

    if (elapsed >= stepDuration) {
      animationState.stepIndex++
      animationState.stepStartTime = now

      if (animationState.stepIndex >= FORWARD_SEQUENCE.length) {
        // Animation complete - consume token from queue if available
        animationState.phase = 'complete'
        animationState.activeComponent = null
        animationState.stepIndex = 0

        // Consume a token from the queue (model already generated it)
        if (tokenState.tokenQueue.length > 0) {
          const nextToken = tokenState.tokenQueue.shift()!
          tokenState.generatedTokens.push(nextToken)
        }

        // Check if we're done
        if (tokenState.generationComplete && tokenState.tokenQueue.length === 0) {
          tokenState.phase = 'complete'
        }
      } else {
        animationState.activeComponent = FORWARD_SEQUENCE[animationState.stepIndex]!
      }
    }
  }

  return commitFrame()
}

// Check if animation should keep running
function shouldContinueAnimation(): boolean {
  // Continue if actively animating forward pass
  if (animationState.phase === 'forward') return true
  // Continue if dragging layers
  if (dragState.active) return true
  // Continue if streaming and waiting for tokens
  if (tokenState.phase === 'streaming' && !tokenState.generationComplete) return true
  // Continue if there are still tokens in queue to display
  if (tokenState.tokenQueue.length > 0) return true
  return false
}

function scheduleRender(): void {
  if (scheduled.value) return
  scheduled.value = true
  requestAnimationFrame(function renderFrame(now) {
    scheduled.value = false
    if (render(now)) scheduleRender()
  })
}

async function startForwardPass(): Promise<void> {
  if (!isModelLoaded()) {
    console.log('Model not loaded yet, waiting for load...')
    const success = await loadModel()
    if (!success) {
      console.error('Failed to load model')
      return
    }
  }

  if (tokenState.phase === 'idle' || tokenState.phase === 'complete') {
    // Get prompt from input
    const prompt = getPrompt()
    if (!prompt) {
      console.log('No prompt entered')
      return
    }

    // Reset state with current prompt
    tokenState.inputText = prompt
    tokenState.generatedTokens = []
    tokenState.tokenQueue = []
    tokenState.generationComplete = false
    tokenState.phase = 'streaming'

    // Reset animation state
    animationState.phase = 'idle'
    animationState.stepIndex = 0
    animationState.activeComponent = null

    // Start generation in background (it will trigger animations)
    runGeneration()
  }

  scheduleRender()
}

function resetAnimation(): void {
  animationState.phase = 'idle'
  animationState.activeComponent = null
  animationState.stepIndex = 0

  // Reset token state
  tokenState.inputText = DEFAULT_PROMPT
  tokenState.generatedTokens = []
  tokenState.tokenQueue = []
  tokenState.generationComplete = false
  tokenState.phase = 'idle'

  // Re-enable and show prompt editor
  setPromptEnabled(true)
  setPromptVisible(true)
  setPrompt(DEFAULT_PROMPT)

  // Reset positions
  for (const pos of layerPositions) {
    pos.x = 0
    pos.y = 0
  }
  embedPosition.x = 0
  embedPosition.y = 0
  headPosition.x = 0
  headPosition.y = 0

  preparedByKey.clear()

  scheduleRender()
}

window.addEventListener('resize', scheduleRender)
btnForward?.addEventListener('click', startForwardPass)
btnReset?.addEventListener('click', resetAnimation)

// Enter key starts generation
promptEditor.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    startForwardPass()
  }
})

// Initialize
await document.fonts.ready
commitFrame()

// Load model in background
loadModel()
