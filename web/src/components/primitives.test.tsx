import { describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { Button, Field, Modal, ProgressBar, Select, TextInput } from './primitives'

describe('Button', () => {
  it('renders its label and fires onClick', async () => {
    const onClick = vi.fn()
    render(<Button onClick={onClick}>Run pipeline</Button>)
    await userEvent.click(screen.getByRole('button', { name: 'Run pipeline' }))
    expect(onClick).toHaveBeenCalledOnce()
  })

  it('does not fire onClick when disabled', async () => {
    const onClick = vi.fn()
    render(<Button onClick={onClick} disabled>Nope</Button>)
    await userEvent.click(screen.getByRole('button', { name: 'Nope' }))
    expect(onClick).not.toHaveBeenCalled()
  })
})

describe('Select', () => {
  it('renders options and reports the chosen value', async () => {
    const onChange = vi.fn()
    render(
      <Select
        value="ja"
        onChange={onChange}
        options={[{ value: 'ja', label: 'Japanese' }, { value: 'ko', label: 'Korean' }]}
      />,
    )
    expect(screen.getByRole('option', { name: 'Japanese' })).toBeInTheDocument()
    await userEvent.selectOptions(screen.getByRole('combobox'), 'ko')
    expect(onChange).toHaveBeenCalledWith('ko')
  })
})

describe('TextInput + Field', () => {
  it('renders a labelled input and emits typed text', async () => {
    const onChange = vi.fn()
    render(
      <Field label="Name" help="required">
        <TextInput value="" onChange={onChange} />
      </Field>,
    )
    expect(screen.getByText('Name')).toBeInTheDocument()
    expect(screen.getByText('required')).toBeInTheDocument()
    await userEvent.type(screen.getByRole('textbox'), 'A')
    expect(onChange).toHaveBeenCalledWith('A')
  })
})

describe('ProgressBar', () => {
  it('renders (clamped) without throwing at extreme values', () => {
    const { container } = render(<ProgressBar value={2} />)
    expect(container.firstChild).toBeTruthy()
  })
})

describe('Modal', () => {
  it('shows its title and closes on Escape', async () => {
    const onClose = vi.fn()
    render(<Modal title="Delete project" onClose={onClose}>body</Modal>)
    expect(screen.getByText('Delete project')).toBeInTheDocument()
    await userEvent.keyboard('{Escape}')
    expect(onClose).toHaveBeenCalled()
  })
})
