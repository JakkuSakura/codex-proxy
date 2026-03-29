use serde::de::{MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::BTreeMap;
use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Number(serde_json::Number),
    String(String),
    Array(Vec<JsonValue>),
    Object(BTreeMap<String, JsonValue>),
}

impl Default for JsonValue {
    fn default() -> Self {
        JsonValue::Null
    }
}

impl JsonValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, JsonValue::Null)
    }
}

impl Serialize for JsonValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            JsonValue::Null => serializer.serialize_unit(),
            JsonValue::Bool(v) => serializer.serialize_bool(*v),
            JsonValue::Number(n) => n.serialize(serializer),
            JsonValue::String(s) => serializer.serialize_str(s),
            JsonValue::Array(arr) => arr.serialize(serializer),
            JsonValue::Object(map) => map.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for JsonValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct JsonValueVisitor;

        impl<'de> Visitor<'de> for JsonValueVisitor {
            type Value = JsonValue;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("any valid JSON value")
            }

            fn visit_unit<E>(self) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(JsonValue::Null)
            }

            fn visit_none<E>(self) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(JsonValue::Null)
            }

            fn visit_bool<E>(self, v: bool) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(JsonValue::Bool(v))
            }

            fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(JsonValue::Number(serde_json::Number::from(v)))
            }

            fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(JsonValue::Number(serde_json::Number::from(v)))
            }

            fn visit_f64<E>(self, v: f64) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                let n = serde_json::Number::from_f64(v)
                    .ok_or_else(|| E::custom("invalid JSON number"))?;
                Ok(JsonValue::Number(n))
            }

            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(JsonValue::String(v.to_string()))
            }

            fn visit_string<E>(self, v: String) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(JsonValue::String(v))
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut items = Vec::new();
                while let Some(v) = seq.next_element::<JsonValue>()? {
                    items.push(v);
                }
                Ok(JsonValue::Array(items))
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut out = BTreeMap::new();
                while let Some((k, v)) = map.next_entry::<String, JsonValue>()? {
                    out.insert(k, v);
                }
                Ok(JsonValue::Object(out))
            }
        }

        deserializer.deserialize_any(JsonValueVisitor)
    }
}
