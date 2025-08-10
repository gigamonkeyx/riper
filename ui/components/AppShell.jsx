import { ThemeProvider, useTheme } from "next-themes";
import { NavigationMenu, NavigationMenuItem, NavigationMenuLink, NavigationMenuList } from "@/components/ui/navigation-menu";
import { Switch } from "@/components/ui/switch";

function ThemeToggle(){
  const { theme, setTheme } = useTheme();
  const checked = theme !== "light"; // dark as default
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-muted-foreground">Light</span>
      <Switch checked={checked} onCheckedChange={(v)=>setTheme(v?"dark":"light")} />
      <span className="text-xs text-muted-foreground">Dark</span>
    </div>
  );
}

import Sidebar from "@/components/Sidebar";
import Rightbar from "@/components/Rightbar";

export default function AppShell({ children, showLeft = true, showRight = true }) {
  return (
    <ThemeProvider attribute="class" defaultTheme="dark" enableSystem>
      <div className="min-h-screen bg-background text-foreground">
        <header className="sticky top-0 z-40 border-b bg-background/80 backdrop-blur">
          <div className="mx-auto max-w-[1400px] px-4 h-14 flex items-center justify-between">
            <div className="font-semibold">Tonasket</div>
            <div className="flex items-center gap-6">
              <NavigationMenu>
                <NavigationMenuList>
                  <NavigationMenuItem>
                    <NavigationMenuLink href="/" className="px-3 py-2">Simulator</NavigationMenuLink>
                  </NavigationMenuItem>
                  <NavigationMenuItem>
                    <NavigationMenuLink href="/funding" className="px-3 py-2">Funding</NavigationMenuLink>
                  </NavigationMenuItem>
                  <NavigationMenuItem>
                    <NavigationMenuLink href="/terminal" className="px-3 py-2">Terminal</NavigationMenuLink>
                  </NavigationMenuItem>
                </NavigationMenuList>
              </NavigationMenu>
              <ThemeToggle />
            </div>
          </div>
        </header>
        <main className="px-4 py-6 mx-auto max-w-[1400px] w-full">
          <div className="grid grid-cols-12 gap-4 items-start">
            {showLeft ? <div className="col-span-12 md:col-span-3 lg:col-span-2"><Sidebar /></div> : null}
            <div className="col-span-12 md:col-span-6 lg:col-span-8 min-w-0">{children}</div>
            {showRight ? <div className="col-span-12 md:col-span-3 lg:col-span-2"><Rightbar /></div> : null}
          </div>
        </main>
      </div>
    </ThemeProvider>
  );
}

