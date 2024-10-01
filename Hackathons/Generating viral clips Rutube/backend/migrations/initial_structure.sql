DROP TABLE IF EXISTS "finished_processed_request";
DROP SEQUENCE IF EXISTS finished_processed_request_id_seq;
CREATE SEQUENCE finished_processed_request_id_seq INCREMENT 1 MINVALUE 1 MAXVALUE 2147483647 START 1 CACHE 1;

CREATE TABLE "public"."finished_processed_request" (
    "id" integer DEFAULT nextval('finished_processed_request_id_seq') NOT NULL,
    "request_id" uuid NOT NULL,
    "filekey" character varying NOT NULL,
    "metrics" character varying NOT NULL,
    "metrics_fields" character varying,
    CONSTRAINT "finished_processed_request_pkey" PRIMARY KEY ("id")
) WITH (oids = false);

DROP TABLE IF EXISTS "request_upload_videos";
DROP SEQUENCE IF EXISTS request_upload_videos_id_seq;
CREATE SEQUENCE request_upload_videos_id_seq INCREMENT 1 MINVALUE 1 MAXVALUE 2147483647 START 1 CACHE 1;

CREATE TABLE "public"."request_upload_videos" (
    "id" integer DEFAULT nextval('request_upload_videos_id_seq') NOT NULL,
    "request_id" uuid NOT NULL,
    "filekey" character varying NOT NULL,
    "is_complete" boolean DEFAULT false NOT NULL,
    CONSTRAINT "request_upload_videos_pkey" PRIMARY KEY ("id")
) WITH (oids = false);
