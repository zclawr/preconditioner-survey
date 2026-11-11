/* Capture actual linear systems without changing the simulation's operators. */
#ifndef KBAND_PETSC_EXPORT_H
#define KBAND_PETSC_EXPORT_H

#include <petscts.h>
#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PETSC_EXPORT_MAX_STEPS 4096

static char     petsc_export_directory[PETSC_MAX_PATH_LEN];
static PetscInt petsc_export_steps[PETSC_EXPORT_MAX_STEPS];
static PetscInt petsc_export_step_count = 0;
static PetscInt petsc_export_every = 1;
static PetscInt petsc_export_ordinal = 0;
static PetscInt petsc_export_count = 0;
static PetscInt petsc_export_last_step = -1;

static PetscErrorCode PetscExportPreSolve(KSP ksp, Vec rhs, Vec solution, void *context)
{
  TS          ts = (TS)context;
  SNES        snes;
  Mat         A, P, exported_A;
  Vec         initial;
  PetscViewer viewer;
  PetscInt    ordinal = petsc_export_ordinal++, step, rows, columns, rhs_size, solution_size;
  PetscInt    snes_iteration, rejections;
  PetscReal   time, dt;
  PetscBool   selected = PETSC_FALSE, nonzero, assembled, has_rows;
  const char *matrix_type, *pc_matrix_type, *exported_type;
  char        path[PETSC_MAX_PATH_LEN], filename[64];
  FILE       *metadata;
  int         written;

  PetscFunctionBeginUser;
  PetscCall(TSGetStepNumber(ts, &step));
  /* Repeated Newton solves and rejected attempts at this step share its label. */
  if (step == petsc_export_last_step) PetscFunctionReturn(PETSC_SUCCESS);
  if (petsc_export_step_count) {
    for (PetscInt i = 0; i < petsc_export_step_count; ++i) {
      if (step == petsc_export_steps[i]) selected = PETSC_TRUE;
    }
  } else {
    selected = (PetscBool)(step % petsc_export_every == 0);
  }
  if (!selected) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(KSPGetOperators(ksp, &A, &P));
  PetscCall(MatGetType(A, &matrix_type));
  PetscCall(MatGetType(P, &pc_matrix_type));
  PetscCall(MatAssembled(A, &assembled));
  PetscCall(MatHasOperation(A, MATOP_GET_ROW, &has_rows));
  PetscCheck(assembled && has_rows, PETSC_COMM_SELF, PETSC_ERR_SUP,
             "Export requires an assembled matrix with accessible entries; operator type is %s. Disable matrix-free options.", matrix_type);
  PetscCall(MatGetSize(A, &rows, &columns));
  PetscCall(VecGetSize(rhs, &rhs_size));
  PetscCall(VecGetSize(solution, &solution_size));
  PetscCheck(rows == columns && rhs_size == rows && solution_size == columns,
             PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Incompatible linear-system dimensions for export");
  PetscCall(TSGetTime(ts, &time));
  PetscCall(TSGetTimeStep(ts, &dt));
  PetscCall(TSGetStepRejections(ts, &rejections));
  PetscCall(TSGetSNES(ts, &snes));
  PetscCall(SNESGetIterationNumber(snes, &snes_iteration));
  PetscCall(KSPGetInitialGuessNonzero(ksp, &nonzero));
  PetscCall(VecDuplicate(solution, &initial));
  if (nonzero) PetscCall(VecCopy(solution, initial));
  else PetscCall(VecSet(initial, 0.0));

  /* Conversion allocates a numeric CSR copy; A and P remain untouched. */
  PetscCall(MatConvert(A, MATAIJ, MAT_INITIAL_MATRIX, &exported_A));
  PetscCall(MatGetType(exported_A, &exported_type));
  PetscCall(PetscSNPrintf(filename, sizeof(filename), "system_%06" PetscInt_FMT ".bin", ordinal));
  PetscCall(PetscSNPrintf(path, sizeof(path), "%s/%s", petsc_export_directory, filename));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_SELF, path, FILE_MODE_WRITE, &viewer));
  PetscCall(PetscViewerBinarySetSkipInfo(viewer, PETSC_TRUE));
  PetscCall(MatView(exported_A, viewer));
  PetscCall(VecView(rhs, viewer));
  PetscCall(VecView(initial, viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(PetscSNPrintf(path, sizeof(path), "%s/systems.jsonl", petsc_export_directory));
  metadata = fopen(path, "a");
  PetscCheck(metadata, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Cannot open export metadata at %s", path);
  written = fprintf(metadata,
                    "{\"ordinal\":%" PetscInt_FMT ",\"step\":%" PetscInt_FMT ","
                    "\"binary_file\":\"%s\",\"ts_time\":%.17g,\"ts_dt\":%.17g,"
                    "\"snes_iteration\":%" PetscInt_FMT ",\"step_rejections\":%" PetscInt_FMT ","
                    "\"rows\":%" PetscInt_FMT ",\"columns\":%" PetscInt_FMT ","
                    "\"operator_type\":\"%s\",\"exported_operator_type\":\"%s\","
                    "\"preconditioning_operator_type\":\"%s\",\"same_operator_and_P\":%s,"
                    "\"native_initial_guess_nonzero\":%s}\n",
                    ordinal, step, filename, (double)time, (double)dt,
                    snes_iteration, rejections, rows, columns, matrix_type,
                    exported_type, pc_matrix_type, A == P ? "true" : "false",
                    nonzero ? "true" : "false");
  PetscCheck(fclose(metadata) == 0 && written >= 0, PETSC_COMM_SELF,
             PETSC_ERR_FILE_WRITE, "Cannot write export metadata at %s", path);
  PetscCall(MatDestroy(&exported_A));
  PetscCall(VecDestroy(&initial));
  petsc_export_last_step = step;
  ++petsc_export_count;
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "PETSC_EXPORT ordinal=%" PetscInt_FMT " step=%" PetscInt_FMT " n=%" PetscInt_FMT "\n", ordinal, step, rows));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscExportAttach(TS ts)
{
  SNES         snes;
  KSP          ksp;
  PetscMPIInt  size;
  PetscBool    directory_set, steps_set;
  char         step_text[PETSC_EXPORT_MAX_STEPS * 24 + 1];
  char        *cursor, *end;
  char         path[PETSC_MAX_PATH_LEN];
  FILE        *metadata;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)ts), &size));
  PetscCheck(size == 1, PETSC_COMM_SELF, PETSC_ERR_SUP, "Linear-system export currently requires one MPI rank");
#if defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Linear-system export requires a real PETSc build");
#endif
  PetscCheck(sizeof(PetscScalar) == sizeof(double), PETSC_COMM_SELF,
             PETSC_ERR_SUP, "Linear-system export requires double-precision PETSc scalars");
  PetscCall(PetscOptionsGetString(NULL, NULL, "-export_dir", petsc_export_directory, sizeof(petsc_export_directory), &directory_set));
  PetscCheck(directory_set && petsc_export_directory[0], PETSC_COMM_SELF,
             PETSC_ERR_USER_INPUT, "Supply -export_dir with an existing output directory");
  PetscCheck(strlen(petsc_export_directory) < sizeof(petsc_export_directory) - 1,
             PETSC_COMM_SELF, PETSC_ERR_USER_INPUT, "Export directory path is too long");
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-export_every", &petsc_export_every, NULL));
  PetscCheck(petsc_export_every > 0, PETSC_COMM_SELF, PETSC_ERR_USER_INPUT, "-export_every must be positive");
  PetscCall(PetscOptionsGetString(NULL, NULL, "-export_steps", step_text, sizeof(step_text), &steps_set));
  petsc_export_step_count = 0;
  if (steps_set) {
    PetscCheck(strlen(step_text) < sizeof(step_text) - 1, PETSC_COMM_SELF,
               PETSC_ERR_USER_INPUT, "-export_steps text is too long");
    cursor = step_text;
    do {
      long long parsed;
      while (isspace((unsigned char)*cursor)) ++cursor;
      PetscCheck(*cursor && petsc_export_step_count < PETSC_EXPORT_MAX_STEPS,
                 PETSC_COMM_SELF, PETSC_ERR_USER_INPUT,
                 "-export_steps must contain 1 to %d nonnegative integers", PETSC_EXPORT_MAX_STEPS);
      errno = 0;
      parsed = strtoll(cursor, &end, 10);
      PetscCheck(end != cursor && errno != ERANGE && parsed >= 0 &&
                 (long long)(PetscInt)parsed == parsed, PETSC_COMM_SELF,
                 PETSC_ERR_USER_INPUT, "Invalid step in -export_steps near '%s'", cursor);
      while (isspace((unsigned char)*end)) ++end;
      PetscCheck(!*end || *end == ',', PETSC_COMM_SELF, PETSC_ERR_USER_INPUT,
                 "Use comma-separated integer values for -export_steps");
      petsc_export_steps[petsc_export_step_count++] = (PetscInt)parsed;
      if (!*end) break;
      cursor = end + 1;
    } while (PETSC_TRUE);
  }
  petsc_export_ordinal = 0;
  petsc_export_count = 0;
  petsc_export_last_step = -1;
  PetscCall(PetscSNPrintf(path, sizeof(path), "%s/systems.jsonl", petsc_export_directory));
  metadata = fopen(path, "w");
  PetscCheck(metadata, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Cannot create export metadata at %s", path);
  PetscCheck(fclose(metadata) == 0, PETSC_COMM_SELF, PETSC_ERR_FILE_WRITE, "Cannot close export metadata at %s", path);
  PetscCall(TSGetSNES(ts, &snes));
  PetscCall(SNESGetKSP(snes, &ksp));
  PetscCall(KSPSetPreSolve(ksp, PetscExportPreSolve, ts));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscExportFinish(TS ts)
{
  TSConvergedReason reason;
  PetscInt          steps, rejections, snes_failures;
  PetscReal         time;
  char              path[PETSC_MAX_PATH_LEN];
  FILE             *metadata;
  int               written;

  PetscFunctionBeginUser;
  PetscCall(TSGetConvergedReason(ts, &reason));
  PetscCall(TSGetStepNumber(ts, &steps));
  PetscCall(TSGetStepRejections(ts, &rejections));
  PetscCall(TSGetSNESFailures(ts, &snes_failures));
  PetscCall(TSGetSolveTime(ts, &time));
  PetscCall(PetscSNPrintf(path, sizeof(path), "%s/run.json", petsc_export_directory));
  metadata = fopen(path, "w");
  PetscCheck(metadata, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Cannot open run metadata at %s", path);
  written = fprintf(metadata,
                    "{\"completed_steps\":%" PetscInt_FMT ",\"ts_converged_reason\":%d,"
                    "\"final_time\":%.17g,\"step_rejections\":%" PetscInt_FMT ","
                    "\"snes_failures\":%" PetscInt_FMT ",\"ksp_calls\":%" PetscInt_FMT ","
                    "\"exported_systems\":%" PetscInt_FMT "}\n",
                    steps, (int)reason, (double)time, rejections, snes_failures,
                    petsc_export_ordinal, petsc_export_count);
  PetscCheck(fclose(metadata) == 0 && written >= 0, PETSC_COMM_SELF,
             PETSC_ERR_FILE_WRITE, "Cannot write run metadata at %s", path);
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "PETSC_EXPORT_COMPLETE steps=%" PetscInt_FMT " reason=%d systems=%" PetscInt_FMT "\n", steps, (int)reason, petsc_export_count));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#endif
