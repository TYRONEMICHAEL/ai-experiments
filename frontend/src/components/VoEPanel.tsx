import { Fact } from "@/lib/types";
import { AnimatePresence, motion } from "framer-motion";

interface VoEPanelProps {
  facts: Fact[];
}

export default function VoEPanel({ facts }: VoEPanelProps) {
  return (
    <div className="flex flex-col h-full w-full">
      <div className="sticky bg-background py-2 top-0 flex flex-row justify-between items-center px-4 text-xs uppercase tracking-wider border-b border-white/20">
        <h3 className="font-mono font-semibold text-sm">Learned Facts (VoE)</h3>
      </div>
      <div className="flex-1 overflow-y-auto px-4 py-2">
        {facts.length === 0 ? (
          <div className="text-center py-8 text-foreground/60 text-sm font-mono">
            No facts learned yet...
          </div>
        ) : (
          <div className="space-y-3">
            <AnimatePresence>
              {facts.map((fact, index) => (
                <motion.div
                  key={fact.id}
                  layout
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.3, delay: index * 0.05 }}
                  className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-lg p-3"
                >
                  <p className="text-sm text-white leading-relaxed mb-2">
                    {fact.text}
                  </p>
                  
                  {fact.tags.length > 0 && (
                    <div className="flex flex-wrap gap-1 mb-2">
                      {fact.tags.map(tag => (
                        <span
                          key={tag}
                          className="text-xs px-2 py-0.5 bg-blue-500/10 text-blue-300 rounded-full font-mono"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  )}
                  
                  <div className="text-xs text-foreground/40 font-mono">
                    {new Date(fact.created_at).toLocaleString([], {
                      month: 'short',
                      day: 'numeric',
                      hour: '2-digit',
                      minute: '2-digit'
                    })}
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        )}
      </div>
    </div>
  );
} 