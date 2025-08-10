# Theming Guide

This document covers theming, shadcn/ui usage, and design patterns for the RIPER UI.

## Overview

The UI uses **shadcn/ui** components with **Tailwind CSS** for consistent theming and design. All components automatically support light/dark mode switching.

## Theme Configuration

### shadcn/ui Setup
Located in `components.json`:
```json
{
  "style": "new-york",
  "rsc": false,
  "tsx": true,
  "tailwind": {
    "baseColor": "slate",
    "cssVariables": true
  }
}
```

### Tailwind Configuration
Key theming features in `tailwind.config.js`:
- CSS variables enabled for dynamic theming
- Dark mode: `class` strategy
- Custom color palette via CSS variables

### Global Styles
`styles/globals.css` defines CSS custom properties:
```css
:root {
  --background: 0 0% 100%;
  --foreground: 222.2 84% 4.9%;
  --muted: 210 40% 96%;
  --muted-foreground: 215.4 16.3% 46.9%;
  /* ... */
}

.dark {
  --background: 222.2 84% 4.9%;
  --foreground: 210 40% 98%;
  /* ... */
}
```

## Design Tokens

Use these Tailwind classes for consistent theming:

### Colors
- `bg-background` - Main background
- `text-foreground` - Primary text
- `text-muted-foreground` - Secondary text
- `bg-muted` - Subtle backgrounds
- `border` - Default borders
- `bg-destructive` - Error states
- `bg-primary` - Primary actions

### Typography
- `text-sm` - Small text (14px)
- `text-base` - Body text (16px)
- `text-lg` - Large text (18px)
- `text-xl` - Extra large (20px)
- `font-medium` - Medium weight
- `font-semibold` - Semi-bold weight

### Spacing
- `p-4` - Padding 16px
- `p-6` - Padding 24px
- `gap-2` - Gap 8px
- `gap-4` - Gap 16px
- `space-y-4` - Vertical spacing 16px

## Component Patterns

### Cards
Use for content panels:
```jsx
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

<Card>
  <CardHeader>
    <CardTitle>Panel Title</CardTitle>
  </CardHeader>
  <CardContent>
    {/* Content */}
  </CardContent>
</Card>
```

### Forms
```jsx
import { Input, Select, Button, Switch } from "@/components/ui/*";

<div className="space-y-4">
  <div>
    <label className="text-xs text-muted-foreground block mb-1">Label</label>
    <Input placeholder="Enter value" />
  </div>
  <Button>Submit</Button>
</div>
```

### Tables
```jsx
import { Table, TableHeader, TableBody, TableHead, TableRow, TableCell } from "@/components/ui/table";

<Table>
  <TableHeader>
    <TableRow>
      <TableHead>Column</TableHead>
    </TableRow>
  </TableHeader>
  <TableBody>
    <TableRow>
      <TableCell>Data</TableCell>
    </TableRow>
  </TableBody>
</Table>
```

### Dialogs
```jsx
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";

<Dialog open={isOpen} onOpenChange={setIsOpen}>
  <DialogContent>
    <DialogHeader>
      <DialogTitle>Dialog Title</DialogTitle>
    </DialogHeader>
    {/* Content */}
  </DialogContent>
</Dialog>
```

## Theme Toggle

The AppShell includes a theme toggle using next-themes:
```jsx
import { useTheme } from "next-themes";
import { Switch } from "@/components/ui/switch";

function ThemeToggle(){
  const { theme, setTheme } = useTheme();
  const checked = theme !== "light";
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-muted-foreground">Light</span>
      <Switch checked={checked} onCheckedChange={(v)=>setTheme(v?"dark":"light")} />
      <span className="text-xs text-muted-foreground">Dark</span>
    </div>
  );
}
```

## Best Practices

### 1. Always Use Design Tokens
❌ Don't use arbitrary colors:
```jsx
<div className="bg-gray-800 text-white">
```

✅ Use design tokens:
```jsx
<div className="bg-background text-foreground">
```

### 2. Consistent Component Usage
❌ Don't mix native and shadcn:
```jsx
<div className="border p-4">
  <button className="bg-blue-500">Action</button>
</div>
```

✅ Use shadcn consistently:
```jsx
<Card>
  <CardContent>
    <Button>Action</Button>
  </CardContent>
</Card>
```

### 3. Proper Semantic Structure
```jsx
<Card>
  <CardHeader>
    <CardTitle>Title</CardTitle>
  </CardHeader>
  <CardContent>
    <div className="space-y-4">
      {/* Content with consistent spacing */}
    </div>
  </CardContent>
</Card>
```

### 4. Responsive Design
```jsx
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
  {/* Responsive grid */}
</div>
```

## Available Components

Current shadcn components in `/components/ui/`:
- `badge` - Status indicators
- `button` - Actions and links
- `card` - Content panels
- `dialog` - Modal overlays
- `input` - Text inputs
- `navigation-menu` - Navigation
- `resizable` - Resizable panels
- `scroll-area` - Custom scrollbars
- `select` - Dropdowns
- `switch` - Toggle controls
- `table` - Data tables
- `tabs` - Tab navigation

## Adding New Components

To add a new shadcn component:
```bash
npx shadcn-ui@latest add [component-name]
```

This will:
1. Download the component to `/components/ui/`
2. Update imports and dependencies
3. Apply the current theme configuration

## Legacy CSS Classes

Some legacy classes remain for specific styling:
- `btn-cyber` - Custom button style
- `crate-card` - Legacy card style (being phased out)
- `crate-panel` - Legacy panel style (being phased out)

**Migrate these to shadcn components when updating pages.**

## Testing Considerations

When writing Playwright tests:
- Use `data-testid` for stable selectors
- Target shadcn components by role: `getByRole('button', { name: 'Submit' })`
- Check component states: `getByRole('switch').getAttribute('aria-checked')`

Example:
```typescript
await expect(page.getByTestId('costs-card')).toBeVisible();
await expect(page.getByRole('button', { name: /Run/i })).toBeVisible();
```
