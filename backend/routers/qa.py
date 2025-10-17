from fastapi import APIRouter, HTTPException
from backend.services.doc_store import DocStore
from backend.services.ai_adapter import query_index, answer_question_from_context, build_index
from backend.models.schemas import QAReq, QAResp
import logging

logger = logging.getLogger("backend.qa")
router = APIRouter()
doc_store = DocStore()

@router.post("/{doc_id}/qa", response_model=QAResp)
async def qa(doc_id: str, req: QAReq):
    try:
        logger.info(f"❓ Q&A request: doc_id={doc_id}, question={req.question}, k={req.k}, page_id={req.page_id}, model={req.model}")
        doc = doc_store.get(doc_id)
        if not doc:
            logger.warning(f"⚠️ Document not found: {doc_id}")
            raise HTTPException(status_code=404, detail="Document not found")
        
        used_contexts = []
        citations = []

        # Build context: if page_id provided, prioritize that page's context
        pcs = doc.get("page_contexts", [])
        page_context = ""
        if req.page_id is not None and 0 <= req.page_id < len(pcs):
            page_context = pcs[req.page_id].get("page_context") or pcs[req.page_id].get("text") or ""
            if page_context and len(page_context.strip()) > 20:
                # Include more context from the current page
                used_contexts.append(f"[Current Slide {req.page_id + 1}]: {page_context[:500]}")
                citations.append({"page_id": req.page_id})

        # If no page context or to enrich, query vector index
        logger.debug(f"Querying vector index with k={req.k}...")
        index = doc.get("index")
        if not index:
            try:
                logger.info("ℹ️ Index missing for doc; rebuilding now...")
                index = build_index(pcs)
                doc["index"] = index
            except Exception as e:
                logger.warning(f"⚠️ Failed to rebuild index: {e}")
        results = query_index(index, req.question, k=req.k)
        logger.debug(f"Found {len(results)} relevant chunks")

        # Merge context from results and build better citations
        total_pages = len(pcs)
        for r in results:
            txt = r.get('text', '') or r.get('page_context', '')
            if txt and len(txt.strip()) > 20:  # Only include substantial content
                pid = r.get('page_id')
                if pid is not None:
                    try:
                        # page_id from vector index is 1-based from ingest.py
                        # Convert to 0-based for frontend, but validate it's in range
                        normalized_pid = int(pid) - 1
                        
                        # Validate page_id is within valid range
                        if 0 <= normalized_pid < total_pages:
                            # Format context with slide number for better LLM understanding
                            formatted_context = f"[Slide {normalized_pid + 1}]: {txt[:500]}"
                            used_contexts.append(formatted_context)
                            citations.append({"page_id": normalized_pid})
                        else:
                            logger.warning(f"Invalid page_id {pid} (normalized: {normalized_pid}) - doc has {total_pages} pages")
                    except Exception as e:
                        logger.warning(f"Failed to process page_id {pid}: {e}")
                        used_contexts.append(txt[:500])  # fallback without slide number

        # Deduplicate citations while preserving order and limit to top 5 most relevant
        seen = set()
        dedup_citations = []
        for c in citations[:5]:  # Limit to top 5 most relevant slides
            pid = c.get("page_id")
            if pid not in seen and pid is not None:
                dedup_citations.append(c)
                seen.add(pid)

        # Build final context string
        if used_contexts:
            context = "\n\n".join(used_contexts)
        else:
            # Fallback to concatenated page contexts
            context = "\n\n".join([(pc.get('page_context') or pc.get('text') or '') for pc in pcs])

        # Generate answer
        logger.debug("Generating answer...")
        answer = answer_question_from_context(context, req.question, req.model)
        logger.info(f"✅ Answer generated: {len(answer)} chars")
        
        # Extract actual citations from the LLM's answer (parse [Slide X] references)
        import re
        actual_citations = []
        citation_pattern = re.compile(r'\[Slide\s+(\d+)\]', re.IGNORECASE)
        cited_slides = set()
        
        for match in citation_pattern.finditer(answer):
            slide_num = int(match.group(1))
            # Convert 1-based slide number to 0-based page_id
            page_id = slide_num - 1
            # Validate it's within range
            if 0 <= page_id < total_pages and page_id not in cited_slides:
                actual_citations.append({"page_id": page_id})
                cited_slides.add(page_id)
        
        logger.debug(f"Extracted {len(actual_citations)} citations from LLM answer")
        
        return {"answer": answer, "citations": actual_citations, "used_contexts": used_contexts}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Q&A failed: {str(e)}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))
