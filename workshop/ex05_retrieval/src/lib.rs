use anyhow::Result;
#[allow(unused_imports)]
use ex03_similarity_solution::cosine_similarity;
use ex04_storage_local_solution::{EmbeddingRecord, EmbeddingStorage};
use std::collections::HashMap;
use uuid::Uuid;
use candle_core::{Tensor, Device};

pub fn add_record(storage: &mut dyn EmbeddingStorage, name: &str, embedding: Vec<f32>) -> Result<String> {
    let record = EmbeddingRecord {
        id: Uuid::new_v4().to_string(),
        name: name.to_string(),
        embedding,
        created_at: chrono::Utc::now(),
        metadata: HashMap::new(),
    };
    
    let id = record.id.clone();
    storage.store_embedding(record)?;
    Ok(id)
}

// Get top-k most similar embeddings to the query
pub fn top_k(storage: &dyn EmbeddingStorage, query: &[f32], k: usize) -> Result<Vec<(EmbeddingRecord, f32)>> {
    !unimplemented!("TODO: implement top_k. The function should return the top k most similar embeddings to the query.")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ex04_storage_local_solution::open_temp_storage;

    // Helper struct to ensure cleanup happens even if test fails
    struct TempFileGuard {
        path: String,
    }
    
    impl Drop for TempFileGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    #[test]
    fn retrieval_returns_best_match_first() -> Result<()> {
        let (mut storage, path) = open_temp_storage()?;
        let _guard = TempFileGuard { path };
        
        let q = vec![1.0, 0.0, 0.0, 0.0];
        // close to q
        add_record(storage.as_mut(), "near", vec![0.9, 0.0, 0.0, 0.0])?;
        // far from q
        add_record(storage.as_mut(), "far", vec![0.0, 1.0, 0.0, 0.0])?;
        let res = top_k(storage.as_ref(), &q, 1)?;
        
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].0.name, "near");
 
        Ok(())
    }

    #[test]
    fn retrieval_sorts_by_similarity_descending() -> Result<()> {
        let (mut storage, path) = open_temp_storage()?;
        let _guard = TempFileGuard { path };
        
        let query = vec![1.0, 0.0, 0.0, 0.0];
        
        // Add multiple records with varying similarity to query
        add_record(storage.as_mut(), "perfect", vec![1.0, 0.0, 0.0, 0.0])?; // cos_sim = 1.0
        add_record(storage.as_mut(), "medium", vec![0.8, 0.8, 0.8, 0.8])?;    // cos_sim ≈ 0.5
        add_record(storage.as_mut(), "good", vec![0.5, 0.5, 0.0, 0.0])?;  // cos_sim ≈ 0.707
        add_record(storage.as_mut(), "poor", vec![0.0, 1.0, 0.0, 0.0])?;    // cos_sim = 0.0
        add_record(storage.as_mut(), "opposite", vec![-1.0, 0.0, 0.0, 0.0])?; // cos_sim = -1.0
        
        let results = top_k(storage.as_ref(), &query, 5)?;
        
        assert_eq!(results.len(), 5);
        assert_eq!(results[0].0.name, "perfect");
        assert_eq!(results[1].0.name, "good");  
        assert_eq!(results[2].0.name, "medium");
        assert_eq!(results[3].0.name, "poor");
        assert_eq!(results[4].0.name, "opposite");
        
        // Check that similarities are in descending order
        for i in 1..results.len() {
            assert!(results[i-1].1 >= results[i].1, "Similarities not in descending order");
        }

        Ok(())
    }

    #[test]
    fn retrieval_respects_k_limit() -> Result<()> {
        let (mut storage, path) = open_temp_storage()?;
        let _guard = TempFileGuard { path };
        
        let query = vec![1.0, 0.0, 0.0];
        
        // Add 5 records
        for i in 0..5 {
            add_record(storage.as_mut(), &format!("record_{}", i), vec![0.1 * i as f32, 0.0, 0.0])?;
        }
        let query_0 = top_k(storage.as_ref(), &query, 0)?;
        let query_2 = top_k(storage.as_ref(), &query, 2)?;
        let query_10 = top_k(storage.as_ref(), &query, 10)?;
        
        // Test different k values
        assert_eq!(query_0.len(), 0);
        assert_eq!(query_2.len(), 2);
        assert_eq!(query_10.len(), 5);
        Ok(())
    }

    #[test]
    fn retrieval_handles_empty_storage() -> Result<()> {
        let (storage, path) = open_temp_storage()?;
        let _guard = TempFileGuard { path };
        
        let query = vec![1.0, 0.0, 0.0];
        
        let results = top_k(storage.as_ref(), &query, 5)?;
        assert_eq!(results.len(), 0);

        Ok(())
    }

}
