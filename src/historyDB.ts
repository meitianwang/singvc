import type { HistoryItem } from "./components/HistoryPanel";

const DB_NAME = "singvc";
const DB_VERSION = 2;
const WAV_STORE = "wav";
const META_STORE = "history";

function open(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(WAV_STORE)) {
        db.createObjectStore(WAV_STORE);
      }
      if (!db.objectStoreNames.contains(META_STORE)) {
        db.createObjectStore(META_STORE);
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

// --- WAV ---

export async function saveWav(id: string, wavB64: string): Promise<void> {
  const db = await open();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(WAV_STORE, "readwrite");
    tx.objectStore(WAV_STORE).put(wavB64, id);
    tx.oncomplete = () => { db.close(); resolve(); };
    tx.onerror = () => { db.close(); reject(tx.error); };
  });
}

export async function loadWav(id: string): Promise<string | undefined> {
  const db = await open();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(WAV_STORE, "readonly");
    const req = tx.objectStore(WAV_STORE).get(id);
    req.onsuccess = () => { db.close(); resolve(req.result ?? undefined); };
    req.onerror = () => { db.close(); reject(req.error); };
  });
}

// --- History metadata ---

const HISTORY_KEY = "items";

export async function saveHistory(items: HistoryItem[]): Promise<void> {
  const db = await open();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(META_STORE, "readwrite");
    tx.objectStore(META_STORE).put(items, HISTORY_KEY);
    tx.oncomplete = () => { db.close(); resolve(); };
    tx.onerror = () => { db.close(); reject(tx.error); };
  });
}

export async function loadHistory(): Promise<HistoryItem[]> {
  const db = await open();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(META_STORE, "readonly");
    const req = tx.objectStore(META_STORE).get(HISTORY_KEY);
    req.onsuccess = () => { db.close(); resolve(req.result ?? []); };
    req.onerror = () => { db.close(); reject(req.error); };
  });
}

export async function addHistoryItem(item: HistoryItem): Promise<void> {
  const items = await loadHistory();
  items.unshift(item);
  await saveHistory(items);
}

// --- Clear all ---

export async function clearAll(): Promise<void> {
  const db = await open();
  return new Promise((resolve, reject) => {
    const tx = db.transaction([WAV_STORE, META_STORE], "readwrite");
    tx.objectStore(WAV_STORE).clear();
    tx.objectStore(META_STORE).clear();
    tx.oncomplete = () => { db.close(); resolve(); };
    tx.onerror = () => { db.close(); reject(tx.error); };
  });
}
