# Development Guide

This guide covers development workflows, testing, and best practices for the RIPER UI.

## Project Structure

```
ui/
├── components/
│   ├── ui/                 # shadcn/ui components
│   ├── AppShell.jsx        # Main layout wrapper
│   ├── Sidebar.jsx         # Navigation sidebar
│   └── Rightbar.jsx        # Right sidebar
├── docs/                   # Documentation
├── pages/
│   ├── _app.js            # Next.js app wrapper
│   ├── index.js           # Dashboard page
│   ├── funding.js         # Funding management
│   └── terminal.jsx       # Terminal interface
├── styles/
│   └── globals.css        # Global styles + shadcn base
├── tests/
│   └── ui.spec.ts         # Playwright E2E tests
├── components.json        # shadcn/ui configuration
├── tailwind.config.js     # Tailwind configuration
└── package.json           # Dependencies
```

## Development Workflow

### 1. Setup
```bash
cd ui
npm install
npm run dev
```

### 2. Adding New Pages
1. Create page file in `/pages/`
2. Import shadcn components needed
3. Wrap in `<AppShell>`
4. Follow established patterns
5. Add tests in `ui.spec.ts`

Example new page:
```jsx
import AppShell from '../components/AppShell';
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

export default function NewPage() {
  return (
    <AppShell>
      <div className="p-6 space-y-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight mb-2">Page Title</h1>
          <p className="text-sm text-muted-foreground">Page description</p>
        </div>
        
        <Card>
          <CardHeader>
            <CardTitle>Section Title</CardTitle>
          </CardHeader>
          <CardContent>
            {/* Content */}
          </CardContent>
        </Card>
      </div>
    </AppShell>
  );
}
```

### 3. Adding shadcn Components
```bash
npx shadcn-ui@latest add [component-name]
```

Available components: https://ui.shadcn.com/docs/components

### 4. Testing
```bash
# Run E2E tests
npm run test:e2e

# Run specific test
npx playwright test --grep "Terminal page"

# Debug mode
npx playwright test --debug
```

## API Integration

### Environment Variables
```bash
# .env.local
NEXT_PUBLIC_API_BASE=http://localhost:8000/api
```

### API Calls Pattern
```jsx
const API_BASE = typeof window !== 'undefined' && process.env.NEXT_PUBLIC_API_BASE
  ? process.env.NEXT_PUBLIC_API_BASE
  : 'http://localhost:8000/api';

// GET request
const fetchData = async () => {
  try {
    const res = await fetch(`${API_BASE}/endpoint`);
    if (!res.ok) throw new Error('Request failed');
    const data = await res.json();
    setData(data);
  } catch (error) {
    setError(error.message);
  }
};

// POST request
const submitData = async (payload) => {
  try {
    const res = await fetch(`${API_BASE}/endpoint`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error('Submit failed');
    const result = await res.json();
    return result;
  } catch (error) {
    console.error('Submit error:', error);
  }
};
```

### Real-time Data (SSE)
```jsx
useEffect(() => {
  const eventSource = new EventSource(`${API_BASE}/stream`);
  
  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      setLiveData(prev => [...prev, data]);
    } catch (error) {
      console.error('SSE parse error:', error);
    }
  };
  
  eventSource.onerror = () => {
    eventSource.close();
  };
  
  return () => eventSource.close();
}, []);
```

## State Management

### Local State
```jsx
import { useState, useEffect } from 'react';

function Component() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    fetchData();
  }, []);
  
  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      // API call
      setData(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };
  
  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error}</div>;
  
  return <div>{/* Render data */}</div>;
}
```

### Form State
```jsx
function FormComponent() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    enabled: false
  });
  
  const handleChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };
  
  const handleSubmit = async () => {
    // Validate and submit
  };
  
  return (
    <form onSubmit={handleSubmit}>
      <Input 
        value={formData.name}
        onChange={(e) => handleChange('name', e.target.value)}
      />
      <Switch 
        checked={formData.enabled}
        onCheckedChange={(v) => handleChange('enabled', v)}
      />
    </form>
  );
}
```

## Testing Best Practices

### Page Object Pattern
```typescript
// tests/pages/funding.ts
export class FundingPage {
  constructor(private page: Page) {}
  
  async goto() {
    await this.page.goto('/funding');
  }
  
  async toggleFirstProgram() {
    const firstRow = this.page.locator('table tbody tr').first();
    const switch = firstRow.getByRole('switch');
    await switch.click();
  }
  
  async expectCostsCardVisible() {
    await expect(this.page.getByTestId('costs-card')).toBeVisible();
  }
}

// In test file
test('funding page works', async ({ page }) => {
  const fundingPage = new FundingPage(page);
  await fundingPage.goto();
  await fundingPage.expectCostsCardVisible();
});
```

### Stable Selectors
```typescript
// ✅ Good - use data-testid for complex components
await page.getByTestId('costs-card');

// ✅ Good - use semantic roles
await page.getByRole('button', { name: /Save/i });
await page.getByRole('switch');

// ✅ Good - use labels for form fields
await page.getByLabel('Email');

// ❌ Avoid - CSS selectors that can break
await page.locator('.btn-cyber');
```

### Async Handling
```typescript
// Wait for API responses
await page.getByRole('button', { name: 'Submit' }).click();
await page.waitForResponse(response => 
  response.url().includes('/api/submit') && response.status() === 200
);

// Wait for elements with timeout
await expect(page.getByText('Success')).toBeVisible({ timeout: 10000 });

// Wait for network idle
await page.waitForLoadState('networkidle');
```

## Performance

### Code Splitting
```jsx
// Dynamic imports for heavy components
import dynamic from 'next/dynamic';

const ChartComponent = dynamic(() => import('../components/Chart'), {
  ssr: false,
  loading: () => <div>Loading chart...</div>
});
```

### Memoization
```jsx
import { useMemo, useCallback } from 'react';

function Component({ data, onUpdate }) {
  // Memoize expensive calculations
  const processedData = useMemo(() => {
    return data.map(item => ({
      ...item,
      calculated: expensiveCalculation(item)
    }));
  }, [data]);
  
  // Memoize callbacks
  const handleUpdate = useCallback((id, value) => {
    onUpdate(id, value);
  }, [onUpdate]);
  
  return (
    <div>
      {processedData.map(item => (
        <Item 
          key={item.id} 
          data={item} 
          onUpdate={handleUpdate}
        />
      ))}
    </div>
  );
}
```

## Debugging

### React DevTools
- Install React Developer Tools browser extension
- Inspect component state and props
- Profile performance

### Network Debugging
```jsx
// Log API calls
const fetchData = async () => {
  console.log('Fetching data from:', `${API_BASE}/endpoint`);
  const res = await fetch(`${API_BASE}/endpoint`);
  console.log('Response status:', res.status);
  const data = await res.json();
  console.log('Response data:', data);
  return data;
};
```

### Error Boundaries
```jsx
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }
  
  static getDerivedStateFromError(error) {
    return { hasError: true };
  }
  
  componentDidCatch(error, errorInfo) {
    console.error('Error caught by boundary:', error, errorInfo);
  }
  
  render() {
    if (this.state.hasError) {
      return <div>Something went wrong.</div>;
    }
    
    return this.props.children;
  }
}
```

## Deployment

### Build Process
```bash
npm run build
npm start
```

### Environment Configuration
```bash
# Production
NEXT_PUBLIC_API_BASE=https://api.production.com

# Staging
NEXT_PUBLIC_API_BASE=https://api.staging.com
```

### Static Export (if needed)
```javascript
// next.config.js
module.exports = {
  output: 'export',
  trailingSlash: true,
  images: {
    unoptimized: true
  }
};
```
