# Component Usage Guide

This guide covers common patterns and best practices for using shadcn/ui components in the RIPER UI.

## Form Components

### Input Fields
```jsx
import { Input } from "@/components/ui/input";

// Basic input
<Input placeholder="Enter value" />

// With label
<div>
  <label className="text-xs text-muted-foreground block mb-1">Field Name</label>
  <Input value={value} onChange={(e) => setValue(e.target.value)} />
</div>

// Controlled input with validation
<Input 
  value={value} 
  onChange={(e) => setValue(e.target.value)}
  className={error ? "border-destructive" : ""}
/>
```

### Select Dropdowns
```jsx
import { Select, SelectTrigger, SelectContent, SelectItem, SelectValue } from "@/components/ui/select";

<Select value={selected} onValueChange={(v) => setSelected(v)}>
  <SelectTrigger className="w-44">
    <SelectValue placeholder="Choose option" />
  </SelectTrigger>
  <SelectContent>
    <SelectItem value="option1">Option 1</SelectItem>
    <SelectItem value="option2">Option 2</SelectItem>
  </SelectContent>
</Select>
```

### Switches and Toggles
```jsx
import { Switch } from "@/components/ui/switch";

// Basic switch
<Switch checked={enabled} onCheckedChange={setEnabled} />

// With label
<label className="flex items-center gap-2">
  <Switch checked={enabled} onCheckedChange={setEnabled} />
  <span>Enable feature</span>
</label>
```

### Buttons
```jsx
import { Button } from "@/components/ui/button";

// Primary button
<Button onClick={handleClick}>Submit</Button>

// Variants
<Button variant="outline">Cancel</Button>
<Button variant="destructive">Delete</Button>
<Button variant="ghost">Link</Button>

// Sizes
<Button size="sm">Small</Button>
<Button size="lg">Large</Button>

// Disabled state
<Button disabled={loading}>
  {loading ? 'Loading...' : 'Submit'}
</Button>
```

## Layout Components

### Cards
```jsx
import { Card, CardHeader, CardTitle, CardContent, CardFooter } from "@/components/ui/card";

// Basic card
<Card>
  <CardContent className="p-6">
    Content here
  </CardContent>
</Card>

// Full card structure
<Card>
  <CardHeader>
    <CardTitle>Card Title</CardTitle>
  </CardHeader>
  <CardContent>
    <div className="space-y-4">
      {/* Content with consistent spacing */}
    </div>
  </CardContent>
  <CardFooter>
    <Button>Action</Button>
  </CardFooter>
</Card>

// Card with custom header
<Card>
  <CardHeader className="flex flex-row items-center justify-between">
    <CardTitle>Title</CardTitle>
    <Button variant="outline" size="sm">Action</Button>
  </CardHeader>
  <CardContent>
    Content
  </CardContent>
</Card>
```

### Tables
```jsx
import { Table, TableHeader, TableBody, TableHead, TableRow, TableCell } from "@/components/ui/table";

<Table>
  <TableHeader>
    <TableRow>
      <TableHead>Name</TableHead>
      <TableHead>Status</TableHead>
      <TableHead>Actions</TableHead>
    </TableRow>
  </TableHeader>
  <TableBody>
    {data.map((item) => (
      <TableRow key={item.id}>
        <TableCell className="font-medium">{item.name}</TableCell>
        <TableCell>
          <Badge variant={item.active ? "default" : "secondary"}>
            {item.active ? "Active" : "Inactive"}
          </Badge>
        </TableCell>
        <TableCell>
          <Button variant="ghost" size="sm">Edit</Button>
        </TableCell>
      </TableRow>
    ))}
  </TableBody>
</Table>
```

### Scroll Areas
```jsx
import { ScrollArea } from "@/components/ui/scroll-area";

// Fixed height scrollable area
<ScrollArea className="h-[400px] p-4">
  <div className="space-y-2">
    {items.map((item) => (
      <div key={item.id} className="p-2 border rounded">
        {item.content}
      </div>
    ))}
  </div>
</ScrollArea>
```

## Interactive Components

### Dialogs
```jsx
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";

// Controlled dialog
<Dialog open={isOpen} onOpenChange={setIsOpen}>
  <DialogContent>
    <DialogHeader>
      <DialogTitle>Confirm Action</DialogTitle>
    </DialogHeader>
    <div className="space-y-4">
      <p>Are you sure you want to continue?</p>
      <div className="flex gap-2 justify-end">
        <Button variant="outline" onClick={() => setIsOpen(false)}>
          Cancel
        </Button>
        <Button onClick={handleConfirm}>
          Confirm
        </Button>
      </div>
    </div>
  </DialogContent>
</Dialog>

// With trigger button
<Dialog>
  <DialogTrigger asChild>
    <Button>Open Dialog</Button>
  </DialogTrigger>
  <DialogContent>
    <DialogHeader>
      <DialogTitle>Dialog Title</DialogTitle>
    </DialogHeader>
    <p>Dialog content</p>
  </DialogContent>
</Dialog>
```

### Tabs
```jsx
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

<Tabs defaultValue="tab1">
  <TabsList>
    <TabsTrigger value="tab1">Tab 1</TabsTrigger>
    <TabsTrigger value="tab2">Tab 2</TabsTrigger>
  </TabsList>
  <TabsContent value="tab1">
    <Card>
      <CardContent className="p-6">
        Tab 1 content
      </CardContent>
    </Card>
  </TabsContent>
  <TabsContent value="tab2">
    <Card>
      <CardContent className="p-6">
        Tab 2 content
      </CardContent>
    </Card>
  </TabsContent>
</Tabs>
```

## Status and Feedback

### Badges
```jsx
import { Badge } from "@/components/ui/badge";

// Status indicators
<Badge>Default</Badge>
<Badge variant="secondary">Secondary</Badge>
<Badge variant="destructive">Error</Badge>
<Badge variant="outline">Outline</Badge>

// Dynamic variants
<Badge variant={status === 'active' ? 'default' : 'secondary'}>
  {status}
</Badge>
```

## Common Patterns

### Form Layout
```jsx
<Card>
  <CardHeader>
    <CardTitle>Form Title</CardTitle>
  </CardHeader>
  <CardContent className="space-y-4">
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div>
        <label className="text-xs text-muted-foreground block mb-1">
          Field 1
        </label>
        <Input />
      </div>
      <div>
        <label className="text-xs text-muted-foreground block mb-1">
          Field 2
        </label>
        <Select>
          <SelectTrigger>
            <SelectValue placeholder="Select..." />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="option1">Option 1</SelectItem>
          </SelectContent>
        </Select>
      </div>
    </div>
    <div className="flex gap-2 justify-end">
      <Button variant="outline">Cancel</Button>
      <Button>Save</Button>
    </div>
  </CardContent>
</Card>
```

### Data Display
```jsx
<Card>
  <CardHeader className="flex flex-row items-center justify-between">
    <CardTitle>Data Table</CardTitle>
    <div className="flex gap-2">
      <Button variant="outline" size="sm">Export</Button>
      <Button size="sm">Add New</Button>
    </div>
  </CardHeader>
  <CardContent className="p-0">
    <Table>
      {/* Table content */}
    </Table>
  </CardContent>
</Card>
```

### Loading States
```jsx
// Button loading
<Button disabled={loading}>
  {loading ? 'Loading...' : 'Submit'}
</Button>

// Content loading
{loading ? (
  <div className="text-center text-muted-foreground py-8">
    Loading...
  </div>
) : (
  <div>{content}</div>
)}

// Empty states
{items.length === 0 && (
  <div className="text-center text-muted-foreground py-8">
    No items found
  </div>
)}
```

## Accessibility

### Proper Labels
```jsx
// Form labels
<div>
  <label htmlFor="email" className="text-xs text-muted-foreground block mb-1">
    Email
  </label>
  <Input id="email" type="email" />
</div>

// Switch labels
<label className="flex items-center gap-2">
  <Switch id="notifications" />
  <span htmlFor="notifications">Enable notifications</span>
</label>
```

### ARIA Attributes
```jsx
// Buttons with descriptions
<Button aria-describedby="help-text">
  Submit
</Button>
<p id="help-text" className="text-xs text-muted-foreground">
  This will save your changes
</p>

// Loading states
<Button disabled={loading} aria-busy={loading}>
  {loading ? 'Saving...' : 'Save'}
</Button>
```

### Focus Management
```jsx
// Auto-focus important inputs
<Input autoFocus placeholder="Search..." />

// Focus trapping in dialogs (handled automatically by shadcn Dialog)
<Dialog>
  <DialogContent>
    {/* Focus is automatically managed */}
  </DialogContent>
</Dialog>
```
