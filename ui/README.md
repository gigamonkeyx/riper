# Tonasket Bakery Simulation UI

Interactive React-based user interface for the Tonasket Bakery Simulation with real-time parameter adjustment and visualization.

Built with **Next.js**, **shadcn/ui**, and **Tailwind CSS** for a modern, accessible, and themeable interface.

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm
- Python 3.8+ with Flask
- Running Tonasket simulation backend

### 1. Start the API Backend
```bash
# From the root directory
python simple_ui_api.py
```
The API will be available at `http://localhost:8000`

### 2. Install UI Dependencies
```bash
cd ui
npm install
```

### 3. Start the Frontend
```bash
npm run dev
```
The UI will be available at `http://localhost:3000`

## 📚 Documentation

- **[Theming Guide](docs/THEMING.md)** - Theme configuration, design tokens, and best practices
- **[Component Guide](docs/COMPONENT_GUIDE.md)** - shadcn/ui component usage patterns
- **[Development Guide](docs/DEVELOPMENT_GUIDE.md)** - Development workflow, testing, and deployment

## 📊 Features

### Interactive Controls
- **🍎 Fruit Capacity**: 5,000-30,000 lbs/year
- **🫙 Mason Jars Output**: 50-500 jars/day
- **📦 Premium Bundles**: 50-500 bundles/day
- **🥩 Meat Processing**: 100-300 lbs/week
- **🍞 Loaf Production**: 500-1,500 loaves/day
- **💰 Wholesale Price**: $2.00-$4.00/loaf
- **🏪 Retail Price**: $4.00-$6.00/loaf

### Visual Displays
- **📊 Revenue Breakdown Chart**: Bar chart showing daily revenue by product category
- **📈 Profit Trends Chart**: Line chart displaying monthly profit trends
- **🎯 Grant Compliance Chart**: Pie chart showing compliance metrics
- **📋 Performance Summary Table**: Key metrics and status indicators
- **💳 Metric Cards**: Real-time revenue, profit, and community impact

### Real-time Updates
- Instant calculation updates when sliders are adjusted
- Live chart and table refreshes
- Responsive design for mobile and desktop

## 🔧 Technical Stack

- **Frontend**: React 18.2.0 + Next.js 14.0.0
- **UI Components**: shadcn/ui with Radix UI primitives
- **Styling**: Tailwind CSS with CSS variables for theming
- **Charts**: Chart.js 4.4.0 + react-chartjs-2 5.2.0
- **Backend**: Flask API with CORS support
- **Testing**: Playwright for E2E testing
- **Theme**: next-themes for dark/light mode
- **Real-time**: Server-sent events for live updates

## 📁 Project Structure

```
ui/
├── components/
│   ├── ui/                   # shadcn/ui components
│   ├── AppShell.jsx          # Main layout wrapper
│   └── Sidebar.jsx           # Navigation components
├── docs/                     # Documentation
│   ├── THEMING.md           # Theme guide
│   ├── COMPONENT_GUIDE.md   # Component patterns
│   └── DEVELOPMENT_GUIDE.md # Development workflow
├── pages/
│   ├── index.js             # Dashboard page
│   ├── funding.js           # Funding management
│   └── terminal.jsx         # Terminal interface
├── styles/
│   └── globals.css          # Global styles + shadcn base
├── tests/
│   └── ui.spec.ts           # Playwright E2E tests
├── components.json          # shadcn/ui configuration
├── tailwind.config.js       # Tailwind configuration
└── package.json             # Dependencies

../
├── simple_ui_api.py          # Flask API backend
└── test_ui_api.py            # API test script
```

## 🧪 Testing

### Test the API
```bash
python test_ui_api.py
```

### Test the UI
1. Start both the API and frontend
2. Open `http://localhost:3000`
3. Adjust sliders and verify real-time updates
4. Check that charts and metrics update correctly

## 📊 API Endpoints

- `GET /api/health` - Health check
- `GET /api/simulation/parameters` - Get slider definitions
- `POST /api/simulation/calculate` - Calculate results from parameters
- `GET /api/simulation/status` - Get simulation status
- `POST /api/simulation/reset` - Reset to defaults

## 🎯 Usage

1. **Adjust Parameters**: Use the sliders to modify simulation parameters
2. **View Results**: Watch real-time updates in charts and metrics
3. **Analyze Impact**: See how changes affect revenue, profit, and community impact
4. **Export Data**: Use built-in export options for charts and data

## 🔧 Development

### Adding New Sliders
1. Update `slider_definitions` in `simple_ui_api.py`
2. Add slider component in `tonasket-sim-ui.jsx`
3. Update calculation logic in the API

### Adding New Charts
1. Import chart type from Chart.js
2. Add chart configuration in the UI component
3. Update API to provide necessary data

## 🚀 Deployment

For production deployment:
1. Build the frontend: `npm run build`
2. Use a production WSGI server for the API
3. Configure proper CORS settings
4. Set up reverse proxy (nginx recommended)

## 📈 Performance

- Real-time calculations with <100ms response time
- Responsive design optimized for all screen sizes
- Efficient chart rendering with Chart.js
- Minimal API calls with client-side caching

## 🎉 Success Metrics

- ✅ 7 interactive parameter sliders
- ✅ 3 chart visualizations + 1 summary table
- ✅ Real-time calculation updates
- ✅ Professional responsive design
- ✅ Full API integration
- ✅ Export capabilities

The UI provides an intuitive interface for exploring different scenarios and understanding the impact of various parameters on the Tonasket Bakery's performance!
