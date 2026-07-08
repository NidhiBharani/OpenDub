import { useEffect } from 'react'
import { useStore } from './state/store'
import { Editor } from './views/Editor'
import { Library } from './views/Library'
import { Settings } from './views/Settings'
import { Toasts } from './components/Toasts'

export function App() {
  const view = useStore((s) => s.view)
  const loadProjects = useStore((s) => s.loadProjects)
  const loadProviders = useStore((s) => s.loadProviders)

  useEffect(() => {
    void loadProjects()
    void loadProviders()
  }, [loadProjects, loadProviders])

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {view === 'library' && <Library />}
      {view === 'editor' && <Editor />}
      {view === 'settings' && <Settings />}
      <Toasts />
    </div>
  )
}
