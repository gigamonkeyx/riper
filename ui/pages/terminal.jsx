import { useEffect, useRef, useState } from 'react';
import AppShell from '../components/AppShell';
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Select, SelectTrigger, SelectContent, SelectItem, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";

const API_BASE = typeof window !== 'undefined' && process.env.NEXT_PUBLIC_API_BASE
  ? process.env.NEXT_PUBLIC_API_BASE
  : 'http://localhost:8000/api';

export default function TerminalPage(){
  const [cmd, setCmd] = useState('status');
  const [args, setArgs] = useState('');
  const [logs, setLogs] = useState([]);
  const [since, setSince] = useState('');
  const [running, setRunning] = useState(false);
  const timerRef = useRef(null);

  useEffect(()=>{
    const es = new EventSource(`${API_BASE}/terminal/stream`);
    es.onmessage = (ev)=>{
      try{ const entry = JSON.parse(ev.data); setLogs(prev=>[...prev, entry]); setSince(entry.ts);
        const el = document.getElementById('term-log'); if (el) el.scrollTop = el.scrollHeight; }catch{}
    };
    es.onerror = ()=>{ es.close(); };
    return ()=> es.close();
  }, []);

  const run = async ()=>{
    setRunning(true);
    try{
      const body = { cmd, args: args? JSON.parse(args): {} };
      const res = await fetch(`${API_BASE}/terminal/run`,{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
      const data = await res.json();
      if (!res.ok) throw new Error(data.error||'error');
    }catch(e){
      // push error into logs visually
    }finally{
      setRunning(false);
    }
  };

  return (
    <AppShell>
      <div className="p-6 space-y-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight mb-2">Terminal</h1>
          <p className="text-sm text-muted-foreground">Execute backend commands and view real-time logs</p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Command Execution</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-end gap-3">
              <div className="min-w-[140px]">
                <label className="text-xs text-muted-foreground block mb-1">Command</label>
                <Select value={cmd} onValueChange={(v)=>setCmd(v)}>
                  <SelectTrigger><SelectValue placeholder="Select command" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="status">status</SelectItem>
                    <SelectItem value="recalc-costs">recalc-costs</SelectItem>
                    <SelectItem value="snapwic">snapwic</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="flex-1">
                <label className="text-xs text-muted-foreground block mb-1">Args (JSON)</label>
                <Input value={args} onChange={e=>setArgs(e.target.value)} placeholder='{"year":3}' />
              </div>
              <Button disabled={running} onClick={run} className="mb-0">
                {running ? 'Running…' : 'Run'}
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card data-testid="terminal-logs">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle>Live Logs</CardTitle>
              <Button variant="outline" size="sm" onClick={()=>{
                const txt = logs.map(l=>`${l.ts} [${l.level}] ${l.msg}`).join('\n');
                navigator.clipboard.writeText(txt);
              }}>Copy All</Button>
            </div>
          </CardHeader>
          <CardContent className="p-0">
            <ScrollArea className="h-[60vh] p-4 font-mono text-sm">
              {logs.map((l,i)=> (
                <div key={i} className="flex items-start gap-2 py-1">
                  <span className="text-xs text-muted-foreground shrink-0">{l.ts}</span>
                  <Badge variant={l.level === 'ERROR' ? 'destructive' : l.level === 'WARN' ? 'secondary' : 'outline'} className="text-xs shrink-0">
                    {l.level}
                  </Badge>
                  <span className="whitespace-pre-wrap break-words">{l.msg}</span>
                </div>
              ))}
              {logs.length === 0 && (
                <div className="text-center text-muted-foreground py-8">
                  No logs yet. Run a command to see output.
                </div>
              )}
            </ScrollArea>
          </CardContent>
        </Card>
      </div>
    </AppShell>
  );
}

