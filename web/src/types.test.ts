import { describe, expect, it } from 'vitest'
import { STAGE_LABELS, STAGE_ORDER, formatTime } from './types'

describe('formatTime', () => {
  it('formats sub-hour durations as m:ss', () => {
    expect(formatTime(29.4)).toBe('0:29')
    expect(formatTime(75)).toBe('1:15')
  })

  it('formats hour+ durations as h:mm:ss', () => {
    expect(formatTime(3661)).toBe('1:01:01')
  })

  it('clamps negatives and non-finite to zero', () => {
    expect(formatTime(-5)).toBe('0:00')
    expect(formatTime(Infinity)).toBe('0:00')
  })

  it('optionally appends milliseconds', () => {
    expect(formatTime(1.234, true)).toBe('0:01.234')
  })

  it('does not lose a millisecond to float error', () => {
    expect(formatTime(6.3, true)).toBe('0:06.300')
    expect(formatTime(5.8, true)).toBe('0:05.800')
  })
})

describe('stage metadata', () => {
  it('has a label for every stage in order', () => {
    expect(STAGE_ORDER).toHaveLength(8)
    for (const key of STAGE_ORDER) expect(STAGE_LABELS[key]).toBeTruthy()
  })
})
