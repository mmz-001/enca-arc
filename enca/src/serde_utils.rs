use serde::{Serialize, de::DeserializeOwned};
use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Write},
    path::Path,
};

pub trait JSONReadWrite {
    fn read_json(path: &str) -> Result<Self, Box<dyn std::error::Error>>
    where
        Self: Sized;

    fn write_json(&self, path: &str) -> Result<(), Box<dyn std::error::Error>>
    where
        Self: Serialize;

    fn load(dir: &str) -> Result<Vec<(String, Self)>, Box<dyn std::error::Error>>
    where
        Self: Sized;
}

impl<I> JSONReadWrite for I
where
    I: DeserializeOwned,
{
    fn read_json(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let result = serde_json::from_reader(reader)?;
        Ok(result)
    }

    fn write_json(&self, path: &str) -> Result<(), Box<dyn std::error::Error>>
    where
        Self: Serialize,
    {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer(&mut writer, self)?;
        writer.flush()?;
        Ok(())
    }

    fn load(dir: &str) -> Result<Vec<(String, Self)>, Box<dyn std::error::Error>>
    where
        Self: Sized,
    {
        let path = Path::new(dir);
        let mut items: Vec<(String, Self)> = Vec::new();

        let entries = fs::read_dir(path)?;

        for entry in entries.filter_map(Result::ok) {
            let entry_path = entry.path();
            if !(entry_path.is_file() && entry_path.extension().and_then(|s| s.to_str()) == Some("json")) {
                continue;
            }

            let file_name = entry_path.file_stem().and_then(|s| s.to_str()).unwrap().to_owned();

            let file = File::open(entry_path)?;
            let mut rdr = BufReader::new(file);
            let item: Self = serde_json::from_reader(&mut rdr).unwrap();
            items.push((file_name, item));
        }

        items.sort_by_key(|item| item.0.clone());

        Ok(items)
    }
}
