/**
 * "Export Snapshot (.json)" and "Import Snapshot (.json)": back up or restore the multiple-testing
 * ledger (trial counts per data window, budget extensions, locked-holdout reveals) and the saved runs
 * with their leaderboards, so they can move between browsers. An import only ever ADDS history –
 * trials and holdout reveals already recorded here are kept – so it cannot be used to forget how
 * many strategies were tried.
 */

import { Download, FileJson, TriangleAlert, Upload } from 'lucide-react';
import { useState, type ChangeEvent } from 'react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { fmtInt } from '@/lib/format';
import { exportSnapshot, importSnapshot, type ImportSummary, type Snapshot } from '@/lib/storage';

/** What a snapshot holds, for the dialogs. */
function contents(snapshot: Snapshot): { trials: number; extensions: number; holdouts: number; runs: number } {
  let trials = 0;
  let extensions = 0;
  let holdouts = 0;
  for (const record of snapshot.ledger) {
    if (record.kind === 'trials') trials += Number((record.data as { count?: number }).count ?? 0);
    else if (record.kind === 'extension') extensions++;
    else holdouts++;
  }
  return { trials, extensions, holdouts, runs: snapshot.runs.length };
}

function download(snapshot: Snapshot): void {
  const blob = new Blob([JSON.stringify(snapshot, null, 1)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `strategyfinder-snapshot-${new Date(snapshot.exportedAt).toISOString().slice(0, 10)}.json`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function ExportDialog(): JSX.Element {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const summary = snapshot ? contents(snapshot) : null;
  return (
    <Dialog onOpenChange={(open) => setSnapshot(open ? exportSnapshot() : null)}>
      <DialogTrigger asChild>
        <Button variant="outline" size="lg" title="Download the trial ledger, holdout log and saved runs as a JSON file">
          <Download /> Export Snapshot (.json)
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <FileJson className="size-5 text-primary" /> Export a snapshot
          </DialogTitle>
          <DialogDescription>
            One JSON file with this browser&apos;s multiple-testing history and saved runs. Import it in another browser to carry on with the same
            trial counts and holdout locks. Cached candles are not included – they are downloaded again.
          </DialogDescription>
        </DialogHeader>
        {summary && (
          <ul className="num space-y-1 text-sm text-secondary-foreground">
            <li>{fmtInt(summary.trials)} trials counted against data windows</li>
            <li>{fmtInt(summary.extensions)} budget extensions</li>
            <li>{fmtInt(summary.holdouts)} locked-holdout reveals</li>
            <li>{fmtInt(summary.runs)} saved runs with their leaderboards</li>
          </ul>
        )}
        <DialogFooter>
          <DialogClose asChild>
            <Button variant="outline">Cancel</Button>
          </DialogClose>
          <Button onClick={() => snapshot && download(snapshot)} disabled={!snapshot}>
            <Download /> Download .json
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function ImportDialog({ onImported }: { onImported: () => void }): JSX.Element {
  const [result, setResult] = useState<ImportSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const onFile = async (event: ChangeEvent<HTMLInputElement>): Promise<void> => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const summary = importSnapshot(await file.text());
      setResult(summary);
      onImported();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'The file could not be read.');
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog
      onOpenChange={(open) => {
        if (!open) return;
        setResult(null);
        setError(null);
      }}
    >
      <DialogTrigger asChild>
        <Button variant="outline" size="lg" title="Merge a snapshot file into this browser's trial ledger, holdout log and saved runs">
          <Upload /> Import Snapshot (.json)
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Upload className="size-5 text-primary" /> Import a snapshot
          </DialogTitle>
          <DialogDescription>
            The file is merged into this browser&apos;s history: imported trials and holdout reveals are added, nothing recorded here is removed. Re-importing
            the same file changes nothing.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-1.5">
          <Label htmlFor="snapshot-file">Snapshot file (.json)</Label>
          <Input id="snapshot-file" type="file" accept="application/json,.json" disabled={busy} onChange={(event) => void onFile(event)} />
        </div>
        {result && (
          <Alert className="border-up/40 bg-up/10 text-up">
            <FileJson />
            <AlertDescription className="num text-current">
              Imported {fmtInt(result.trials)} trial records, {fmtInt(result.extensions)} budget extensions, {fmtInt(result.holdouts)} holdout reveals and{' '}
              {fmtInt(result.runs)} saved runs{result.skipped > 0 ? ` (${fmtInt(result.skipped)} malformed records skipped)` : ''}.
            </AlertDescription>
          </Alert>
        )}
        {error && (
          <Alert className="border-down/40 bg-down/10 text-down">
            <TriangleAlert />
            <AlertDescription className="text-current">{error}</AlertDescription>
          </Alert>
        )}
        <DialogFooter>
          <DialogClose asChild>
            <Button variant="outline">Close</Button>
          </DialogClose>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

/** Both snapshot controls, side by side in the top action bar. */
export default function SnapshotDialogs({ onImported }: { onImported: () => void }): JSX.Element {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <ExportDialog />
      <ImportDialog onImported={onImported} />
    </div>
  );
}
