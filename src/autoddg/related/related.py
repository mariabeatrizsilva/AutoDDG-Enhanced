"""Related work profiler for extracting dataset context from research papers."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Dict, List, Any, Literal

import yaml
from beartype import beartype
from openai import OpenAI
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


@beartype
class RelatedWorkProfiler:
    """
    Extracts dataset context and information from related research papers.
   
    This profiler analyzes research papers (PDFs) to extract contextual information
    about datasets, including their characteristics, usage, and provenance.
    """
   
    def __init__(
        self,
        client: OpenAI,
        model_name: str = "gpt-4o-mini",
        prompts_config: Optional[dict] = None,
    ) -> None:
        """
        Initialize the RelatedWorkProfiler.
       
        Args:
            client: OpenAI client instance for LLM calls
            model_name: Name of the model to use for extraction
            prompts_config: Dictionary containing prompts configuration. If None, loads from prompts.yaml
        """
        self.client = client
        self.model_name = model_name
       
        # Load prompts from config
        if prompts_config is None:
            prompts_config = self._load_prompts_config()
       
        self.prompts = prompts_config.get("related_work_extraction", {})
        self.default_extraction_prompt = self.prompts.get("default_prompt", "")
        self.system_message = self.prompts.get("system_message", "You are an expert academic research assistant.")
   
    def _load_prompts_config(self) -> dict:
        """
        Load prompts configuration from prompts.yaml file.
       
        Returns:
            Dictionary containing prompts configuration
        """
        # Try to find prompts.yaml in the autoddg package
        try:
            from importlib.resources import files
            prompts_path = files("autoddg.configurations").joinpath("prompts.yaml")
            with prompts_path.open("r") as f:
                return yaml.safe_load(f)
        except (ImportError, FileNotFoundError):
            # Fallback: try relative path
            # We're in autoddg/related/related.py, need to go to autoddg/configurations/
            current_dir = Path(__file__).parent  # autoddg/related/
            autoddg_dir = current_dir.parent      # autoddg/
            prompts_path = autoddg_dir / "configurations" / "prompts.yaml"
           
            if prompts_path.exists():
                with open(prompts_path, "r") as f:
                    return yaml.safe_load(f)
            else:
                # Return empty dict if no config found
                print(f"Warning: prompts.yaml not found at {prompts_path}, using empty config")
                return {}
   
    @beartype
    def remove_references_section(self, paper_text: str) -> str:
        """
        Remove the references/bibliography section from the paper text.
       
        This prevents the LLM from confusing cited datasets with the paper's actual usage.
       
        Args:
            paper_text: Full text of the research paper
           
        Returns:
            Paper text with references section removed
        """
        # Common reference section headers (case-insensitive)
        ref_patterns = [
            r'\n\s*REFERENCES\s*\n',
            r'\n\s*References\s*\n',
            r'\n\s*BIBLIOGRAPHY\s*\n',
            r'\n\s*Bibliography\s*\n',
            r'\n\s*WORKS CITED\s*\n',
            r'\n\s*Works Cited\s*\n',
            r'\n\s*LITERATURE CITED\s*\n',
            r'\n\s*Literature Cited\s*\n',
        ]
       
        # Find the earliest occurrence of any reference header
        earliest_ref_position = len(paper_text)
        matched_pattern = None
       
        for pattern in ref_patterns:
            match = re.search(pattern, paper_text, re.IGNORECASE)
            if match and match.start() < earliest_ref_position:
                earliest_ref_position = match.start()
                matched_pattern = pattern
       
        # If we found a reference section, truncate there
        if matched_pattern is not None:
            paper_text_cleaned = paper_text[:earliest_ref_position]
            removed_length = len(paper_text) - len(paper_text_cleaned)
            print(f"Removed references section ({removed_length} characters, {removed_length/len(paper_text)*100:.1f}% of paper)")
            return paper_text_cleaned
        else:
            print("No references section detected, using full text")
            return paper_text
   
    @beartype
    def chunk_text(
        self,
        paper_text: str,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
    ) -> list[str]:
        """
        Splits the full paper text into context-preserving chunks.
       
        Args:
            paper_text: The full text content of the research paper.
            chunk_size: The desired maximum size of each chunk (in characters).
            chunk_overlap: The number of characters to overlap between adjacent chunks.
           
        Returns:
            A list of text strings (chunks).
        """
        # Use standard academic separators to preserve paragraphs and sentences
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",  # Try to split by paragraph first
                "\n",    # Then by newline
                " ",     # Then by space
                "",      # Fallback by character
            ],
            length_function=len,
            is_separator_regex=False,
        )
       
        chunks = splitter.split_text(paper_text)
        print(f"Original text split into {len(chunks)} chunks.")
        return chunks
    
    def chunk_text_with_indices(self, text: str, chunk_size: int, chunk_overlap: int) -> List[tuple]:
        """
        Returns a list of tuples: (text_content, start_index, end_index)
        """
        if not text:
            return []
            
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk_text = text[start:end]
            
            # Save the tuple (Text, Start, End)
            chunks.append((chunk_text, start, end))
            
            # Stop if we hit the end
            if end == text_len:
                break
                
            # Move the window forward (subtract overlap to go back a bit)
            start = end - chunk_overlap
            
        return chunks
   
    def merge_chunks(self, relevant_chunks: List[tuple]) -> List[str]:
        """
        Inputs: List of (text, start, end) tuples that were marked 'Relevant'.
        Returns: List of clean string blocks with overlaps resolved.
        """
        if not relevant_chunks:
            return []

        # 1. Sort by start index to ensure order
        sorted_chunks = sorted(relevant_chunks, key=lambda x: x[1])
        
        merged_blocks = []
        
        # Initialize with the first chunk
        current_text, current_start, current_end = sorted_chunks[0]

        for i in range(1, len(sorted_chunks)):
            next_text, next_start, next_end = sorted_chunks[i]

            # CHECK: Are they neighbors? (Does the next one start before the current one ends?)
            if next_start < current_end:
                # They overlap! Calculate the non-overlapping part of the new chunk
                # The 'overlap_len' is how far back the new chunk starts relative to the old end
                # Actually, simpler: we just want the text from current_end onwards
                
                # How much of the next chunk is actually NEW?
                # It starts at next_start, but we already have data up to current_end.
                # So we cut off the first (current_end - next_start) characters.
                start_cut_index = current_end - next_start
                
                if start_cut_index < len(next_text):
                    # Append only the new part
                    current_text += next_text[start_cut_index:]
                    current_end = next_end
            else:
                # There is a gap. Close the current block and start a new one.
                merged_blocks.append(current_text)
                current_text = next_text
                current_start = next_start
                current_end = next_end

        # Don't forget the last block
        merged_blocks.append(current_text)
        
        return merged_blocks

    def find_anchor_chunks(
        self,
        chunks: List[str],
        dataset_name: str,
        min_tokens_to_match: int = 2
    ) -> List[int]:
        """
        Searches a list of text chunks for references to a given dataset name.

        This function implements a semi-broad search strategy:
        1. Splits the dataset name into key tokens (excluding common words).
        2. Requires a minimum number of these key tokens to be present in a chunk.
        3. The search is case-insensitive.

        Returns:
            A list of IDs or indices of the chunks that contain enough matching tokens.
        """
        # 1. Pre-process the dataset name to get key search terms
        # Define common stop words to ignore (can be expanded)
        stop_words = {'the', 'a', 'an', 'database', 'data', 'of', 'and', 'for', 'in', 'to', 'with'}
       
        # Split the name into tokens, filter out stop words, and convert to lowercase
        key_tokens = set(
            re.findall(r'\b\w+\b', dataset_name.lower())
        ) - stop_words

        if not key_tokens:
            print("Warning: Dataset name contains only stop words after filtering. Cannot perform robust search.")
            # Fallback to searching the full, non-processed name
            key_tokens = {dataset_name.lower()}
            # If we use the full name, we must match at least 1 token
            min_tokens_to_match = 1
       
        # Ensure min_tokens_to_match is not more than the number of key tokens
        min_tokens_to_match = min(min_tokens_to_match, len(key_tokens))
       
        # If a short name like 'FluPRINT' is used, require matching all tokens
        if len(key_tokens) < min_tokens_to_match:
            min_tokens_to_match = len(key_tokens)
       
        # 2. Search each chunk
        anchor_chunk_ids = []
       
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.lower()
           
            # Identify which key tokens are present in the current chunk
            matched_tokens = 0
           
            for token in key_tokens:
                # Check for the token as a whole word boundary match
                if re.search(r'\b' + re.escape(token) + r'\b', chunk_text):
                    matched_tokens += 1
                # print(f"looking for {token} in {chunk_text} \n")
                   
            # 3. Apply the matching threshold
            if matched_tokens >= min_tokens_to_match:
                # We use 'id' if available, otherwise the index 'i'
                chunk_identifier = i
                anchor_chunk_ids.append(chunk_identifier)

        return anchor_chunk_ids
   
    def get_logical_context_blocks(
        self,
        all_chunks: List[str],
        anchor_chunk_ids: List[int],
        context_window_size: int = 2
    ) -> List[str]:
        """
        Creates coherent, logical context blocks by merging adjacent anchor chunks
        and expanding the context window around non-adjacent ones.
        """
       
        # 1. Sort and ensure uniqueness
        sorted_anchor_ids = sorted(list(set(anchor_chunk_ids)))
       
        # 2. Identify all indices to include in the final context
        context_indices_to_include = set()
        num_chunks = len(all_chunks)
       
        # Iterate through anchor chunks to apply merging/expansion
        for anchor_id in sorted_anchor_ids:
           
            # Skip if this chunk is already part of a previous block's context
            if anchor_id in context_indices_to_include:
                continue
               
            # Determine the boundaries for this logical block (expansion)
            # Start by expanding the context around the anchor
            start_index = max(0, anchor_id - context_window_size)
            end_index = min(num_chunks - 1, anchor_id + context_window_size)

            # Extend the end_index if the next chunks are also anchors (merging)
            current_id = anchor_id + 1
            while current_id <= end_index and current_id in sorted_anchor_ids:
                # Keep extending the block to include this anchor too
                end_index = min(num_chunks - 1, current_id + context_window_size)
                current_id += 1
           
            # Add all indices in this logical block to our set
            for idx in range(start_index, end_index + 1):
                context_indices_to_include.add(idx)
       
        # 3. Build the final logical blocks from consecutive indices
        sorted_indices = sorted(list(context_indices_to_include))
        logical_blocks = []
       
        if not sorted_indices:
            return []
       
        # Merge consecutive indices into blocks of text
        current_block_start = sorted_indices[0]
        current_block_end = sorted_indices[0]
       
        for idx in sorted_indices[1:]:
            if idx == current_block_end + 1:
                # Continue the current block
                current_block_end = idx
            else:
                # Start a new block
                # Finalize the previous block
                block_text = " ".join(all_chunks[current_block_start : current_block_end + 1])
                logical_blocks.append(block_text)
               
                current_block_start = idx
                current_block_end = idx
       
        # Add the last block
        block_text = " ".join(all_chunks[current_block_start : current_block_end + 1])
        logical_blocks.append(block_text)
       
        print(f"Created {len(logical_blocks)} logical context blocks from {len(sorted_anchor_ids)} anchor chunks")
       
        return logical_blocks
   
    @beartype
    def extract_text_from_pdf(
        self,
        pdf_path: str,
        max_pages: Optional[int] = None
    ) -> str:
        """
        Extract text content from a PDF file.
       
        Args:
            pdf_path: Path to the PDF file
            max_pages: Optional limit on number of pages to extract
           
        Returns:
            Extracted text as a single string
        """
        try:
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)
            pages_to_extract = min(max_pages, total_pages) if max_pages else total_pages
           
            print(f"Extracting text from {pages_to_extract} pages...")
           
            text_parts = []
            for page_num in range(pages_to_extract):
                page = reader.pages[page_num]
                text_parts.append(page.extract_text())
           
            full_text = "\n".join(text_parts)
            print(f"Extracted {len(full_text)} characters from PDF")
           
            return full_text
           
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {e}")
   
    @beartype
    def _extract_profile_from_full_text(
        self,
        paper_text: str,
        dataset_name: str,
        extraction_prompt: Optional[str] = None,
    ) -> dict:
        """
        Extract related work profile directly from full paper text using LLM.
       
        Args:
            paper_text: Full text content of the paper
            dataset_name: Name of the dataset to focus extraction on
            extraction_prompt: Custom extraction prompt. If None, uses default.
           
        Returns:
            Dictionary containing the related work profile with keys:
                - summary: Extracted summary about the datraset
                - dataset_name: Name of the dataset
                - source_length: Character count of source paper
        """
        # Use custom prompt if provided, otherwise use default
        prompt_template = extraction_prompt if extraction_prompt else self.default_extraction_prompt
       
        # Format the prompt with the paper text and dataset name
        formatted_prompt = prompt_template.format(
            paper_text=paper_text,
            dataset_name=dataset_name
        )
       
        print(f"Extracting profile for dataset: {dataset_name}")
        print(f"Sending {len(formatted_prompt)} characters to LLM...")
       
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": self.system_message
                    },
                    {
                        "role": "user",
                        "content": formatted_prompt
                    }
                ],
                temperature=0.1,
                # response_format={"type": "json_object"}
            )
           
            summary = response.choices[0].message.content.strip()
           
            print(f"Successfully extracted profile ({len(summary)} characters)")
           
            return {
                "summary": summary,
                "dataset_name": dataset_name,
                "source_length": len(paper_text)
            }
           
        except Exception as e:
            raise Exception(f"Error calling LLM for extraction: {e}")
       
    @beartype
    def _extract_profile_from_context(
        self,
        context_blocks: List[str],  # New parameter: list of relevant context blocks
        dataset_name: str,
        extraction_prompt: Optional[str] = None,
    ) -> dict:
        """
        Extract related work profile from selected context blocks using LLM.
        """
       
        # 1. Combine all logical context blocks into a single string
        # Use a clear separator so the LLM knows where one block ends and the next begins
        combined_context = "\n\n--- LOGICAL BLOCK SEPARATOR ---\n\n".join(context_blocks)
       
        # Use custom prompt if provided, otherwise use default
        prompt_template = extraction_prompt if extraction_prompt else self.default_extraction_prompt
       
        # Format the prompt with the combined context and dataset name
        formatted_prompt = prompt_template.format(
            paper_text=combined_context,  # paper_text now refers to the combined, relevant context
            dataset_name=dataset_name
        )
       
        print(f"Extracting profile for dataset: {dataset_name}")
        print(f"Sending {len(formatted_prompt)} characters of CONTEXT to LLM...")
       
        # Call the LLM (rest of the code remains the same)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": self.system_message
                    },
                    {
                        "role": "user",
                        "content": formatted_prompt
                    }
                ],
                temperature=0.1,
                # response_format={"type": "json_object"}
            )
           
            summary = response.choices[0].message.content.strip()
           
            print(f"Successfully extracted profile ({len(summary)} characters)")
           
            return {
                "summary": summary,
                "dataset_name": dataset_name,
                "source_length": len(combined_context) # Source length is now the context size
            }
           
        except Exception as e:
            raise Exception(f"Error calling LLM for extraction: {e}")
   
    @beartype
    def analyze_paper(
        self,
        pdf_path: str,
        dataset_name: str,
        extraction_prompt: Optional[str] = None,
        max_pages: Optional[int] = None,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        context_window_size: int = 3,
        extraction_strategy: Literal["full", "keyword", "llm_context"] = "keyword"    ) -> dict:
        """
        Complete pipeline: Extract text from PDF and generate related work profile.
       
        Args:
            pdf_path: Path to the PDF file
            dataset_name: Name of the dataset to focus extraction on
            extraction_prompt: Custom extraction prompt. If None, uses default.
            max_pages: Optional limit on number of pages to extract
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks
            context_window_size: Number of chunks before/after anchor to include
            extraction_strategy: If True, use keyword-based search (faster, cheaper).
                               If False, use LLM-based relevance scoring (slower, more expensive).
           
        Returns:
            Dictionary containing the related work profile
        """
        # Step 1: Extract text from PDF
        paper_text = self.extract_text_from_pdf(pdf_path, max_pages=max_pages)
       
        # Step 2: Remove references section (NEW!)
        paper_text = self.remove_references_section(paper_text)
       
        logical_context_blocks = []

        all_chunks_with_indices = self.chunk_text_with_indices(
            paper_text, 
            chunk_size, 
            chunk_overlap
        )

        relevant_chunks_with_indices = []
        # Step 3: Branch based on strategy
        if extraction_strategy == "full":
            print(f"Strategy 'full': Using entire text ({len(paper_text)} chars).")
            # Treat the full text as a single context block
            logical_context_blocks = [paper_text]

        else:
            # Both 'keyword' and 'llm_context' require chunking first
            original_chunks = self.chunk_text(
                paper_text=paper_text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            if extraction_strategy == "keyword":
                print("Strategy 'keyword': Searching for tokens...")
                
                # 1. Prepare data for your search function
                # Extract just the text string from each tuple so find_anchor_chunks is happy
                plain_text_chunks = [chunk[0] for chunk in all_chunks_with_indices]
                
                # 2. Get the indices of the "hits"
                # This returns integers like [4, 5, 20, 21]
                anchor_indices = self.find_anchor_chunks(
                    chunks=plain_text_chunks, 
                    dataset_name=dataset_name, 
                    min_tokens_to_match=1
                )
                
                # 3. Expand Context (Apply the Window)
                # Grab neighbors around the anchors
                indices_to_keep = set()
                num_chunks = len(all_chunks_with_indices)
                
                for anchor_idx in anchor_indices:
                    window_start = max(0, anchor_idx - context_window_size)
                    window_end = min(num_chunks, anchor_idx + context_window_size + 1)
                    indices_to_keep.update(range(window_start, window_end))

                # 4. Map indices back to the full Tuple objects (Text, Start, End)
                sorted_indices = sorted(list(indices_to_keep))
                relevant_chunks_with_indices = [all_chunks_with_indices[i] for i in sorted_indices]
            
        # ... Then proceed to self.merge_chunks(relevant_chunks_with_indices)
            elif extraction_strategy == "llm_context":
                print("Scoring chunks...")
                for chunk_tuple in all_chunks_with_indices:
                    text_content = chunk_tuple[0] # Grab just the string for the LLM
                    
                    # Pass the string to your existing scorer
                    if self.score_chunk_relevance(text_content, dataset_name):
                        relevant_chunks_with_indices.append(chunk_tuple)

            # Merge the relevant chunks before creating context
            # This returns a list of clean strings (no overlaps, no extra separators for neighbors)
            final_logical_blocks = self.merge_chunks(relevant_chunks_with_indices)

            current_context_length = sum(len(block) for block in final_logical_blocks)

            # If we found nothing, OR if what we found is suspiciously short (< 1000 chars), use full text
            if not final_logical_blocks or current_context_length < 1000:
                print(f"Warning: Context too short ({current_context_length} chars). Falling back to FULL TEXT.")
                # We overwrite the blocks list with the single full text block
                final_logical_blocks = [paper_text]
            # 3. Extract Profile
            profile = self._extract_profile_from_context(
                context_blocks=final_logical_blocks, # Now sending clean merged text
                dataset_name=dataset_name,
                extraction_prompt=extraction_prompt
            )
       
        profile["full_source_length"] = len(paper_text)
        profile["num_context_blocks"] = len(logical_context_blocks)
        profile["extraction_strategy"] = extraction_strategy
       
        return profile
   
    # LLM-based relevance
    def score_chunk_relevance(self, chunk_text: str, dataset_name: str) -> bool:
        """
        Uses LLM to determine if a chunk is semantically relevant to the dataset.
       
        NOTE: This is expensive and may still include references. Consider using
        keyword-based search (find_anchor_chunks) instead.
        """
       
        scoring_model = self.model_name

        prompt = f"""
        Analyze the following text chunk from a research paper. The focus is on the dataset named '{dataset_name}'.
       
        Chunk:
        ---
        {chunk_text}
        ---
       
        Is this chunk semantically relevant to the dataset? Relevance means it discusses the dataset's use, characteristics, results, or limitations. It is NOT relevant if it is a list of references, acknowledgments, or a general background statement.
       
        Respond with only a single word: YES or NO.
        """
       
        try:
            response = self.client.chat.completions.create(
                model=scoring_model,
                messages=[
                    {"role": "system", "content": "You are a text relevance classifier. Respond only with 'YES' or 'NO'."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
            )
           
            response_content = response.choices[0].message.content.strip().upper()
           
            return response_content == "YES"
           
        except Exception as e:
            print(f"Error scoring chunk: {e}. Defaulting to non-relevant.")
            return False
