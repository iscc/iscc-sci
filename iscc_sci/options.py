from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


__all__ = [
    "SciOptions",
    "sci_opts",
]


load_dotenv()


class SciOptions(BaseSettings):
    bits: int = Field(
        64,
        description="ISCC_SCI_BITS - Default bit-length of generated Semantic Text-Image in bits",
        ge=32,
        le=256,
        multiple_of=32,
    )

    embedding: bool = Field(
        False, description="ISCC_SCI_EMBEDDING - Include image embedding vector"
    )

    precision: int = Field(
        8, description="ISCC_SCI_PRECISION - Max fractional digits for embeddings (default 8)"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="ISCC_SCI_",
        extra="ignore",
        validate_assignment=True,
    )

    def override(self, update=None):
        # type: (dict|None) -> SciOptions
        """Returns an updated and validated deep copy of the current settings instance."""

        update = update or {}  # sets {} if update is None

        opts = self.model_copy(deep=True)
        # We need to update fields individually so validation gets triggered
        for field, value in update.items():
            setattr(opts, field, value)
        return opts


sci_opts = SciOptions()
